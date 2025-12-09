import cv2
import math
import networkx as nx
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

class OnlineObjectDetector:
    """
    Wraps YOLOv8. Automatically downloads weights if not present.
    """
    def __init__(self, model_name='yolov8m.pt'):
        # 'yolov8m.pt' will automatically download from Ultralytics
        print(f"[Detector] Loading {model_name}...")
        self.model = YOLO(model_name)

    def detect(self, image, conf_threshold=0.3):
        # Convert PIL Image to numpy for YOLO
        import numpy as np
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Run inference
        results = self.model(img_bgr, conf=conf_threshold, verbose=False)[0]
        
        objects = []
        names = results.names
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls)
            conf = float(box.conf)
            
            obj = {
                "id": len(objects),
                "label": names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2]
            }
            objects.append(obj)
            
        return objects, results.plot() # Returns objects + annotated image (BGR)

class SceneGraphBuilder:
    """
    Constructs a scene graph using spatial heuristics.
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, objects, width, height):
        self.graph.clear()
        for obj in objects:
            self.graph.add_node(obj['id'], label=obj['label'])

        for i, objA in enumerate(objects):
            for j, objB in enumerate(objects):
                if i == j: continue
                
                dist = math.hypot(objA['center'][0] - objB['center'][0], 
                                  objA['center'][1] - objB['center'][1])
                
                # Logic: "Person interacting with X"
                if objA['label'] == 'person' and dist < (width * 0.15):
                    self.graph.add_edge(i, j, relation="interacting_with")
                
                # Logic: "Near"
                elif dist < (width * 0.2):
                    self.graph.add_edge(i, j, relation="near")

        return self.graph

class OnlineCaptioner:
    """
    Wraps BLIP. Fetches model from Hugging Face Hub.
    """
    def __init__(self, model_repo='Salesforce/blip-image-captioning-base'):
        # Force CPU if no GPU available (Streamlit Cloud is usually CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Captioner] Loading from {model_repo} to {self.device}...")
        
        self.processor = BlipProcessor.from_pretrained(model_repo)
        self.model = BlipForConditionalGeneration.from_pretrained(model_repo).to(self.device)

    def generate(self, image):
        # Image is expected to be a PIL Image
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption