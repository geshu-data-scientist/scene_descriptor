import streamlit as st
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Import our backend classes
from utils_o import OnlineObjectDetector, SceneGraphBuilder, OnlineCaptioner

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="AI Scene Understanding")

# --- CACHED MODEL LOADER ---
@st.cache_resource
def load_models():
    """
    Initialize models. This runs ONCE per session.
    Downloads weights automatically.
    """
    # Use smaller models for Cloud Stability (avoid OOM errors)
    # Change 'yolov8n.pt' to 'yolov8x.pt' if you have a powerful server
    detector = OnlineObjectDetector(model_name='yolov8m.pt') 
    
    # Use 'Salesforce/blip-image-captioning-large' if you have >16GB RAM
    captioner = OnlineCaptioner(model_repo='Salesforce/blip-image-captioning-base')
    
    graph_builder = SceneGraphBuilder()
    return detector, captioner, graph_builder

# --- MAIN APP ---
def main():
    st.title("🧠 AI Scene Understanding System")
    st.markdown("Generates **Captions**, **Object Detection**, and **Scene Graphs** from images.")

    # Load Models (Show spinner on first run)
    with st.spinner("Downloading and Loading Models... (This may take a minute first time)"):
        try:
            detector, captioner, graph_builder = load_models()
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return

    # Sidebar
    st.sidebar.header("Settings")
    conf_thresh = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.3)

    # File Upload
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        # 1. Read Image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)

        if st.button("Analyze Scene"):
            with st.spinner("Processing..."):
                # A. Captioning
                caption = captioner.generate(image)
                
                # B. Detection
                objects, annotated_img_bgr = detector.detect(image, conf_threshold=conf_thresh)
                annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = annotated_img_rgb.shape

                # C. Graph
                G = graph_builder.build_graph(objects, w, h)

            # --- DISPLAY RESULTS ---
            st.success("Analysis Complete!")
            
            # 1. Text Summary
            st.subheader("📝 Generated Description")
            st.info(f"**Caption:** {caption}")

            # 2. Visuals
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.write("### Object Detection")
                st.image(annotated_img_rgb, caption=f"Detected {len(objects)} Objects", use_container_width=True)

            with res_col2:
                st.write("### Scene Graph")
                if G.number_of_nodes() > 0:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    pos = nx.spring_layout(G, k=0.8)
                    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                            node_size=1500, font_weight='bold', ax=ax)
                    edge_labels = nx.get_edge_attributes(G, 'relation')
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
                    st.pyplot(fig)
                else:
                    st.write("No relationships detected.")

            # 3. Text Report
            with st.expander("View Detailed Report"):
                st.write(f"**Objects Found:** {len(objects)}")
                for obj in objects:
                    st.write(f"- {obj['label']} ({obj['confidence']:.2f})")
                st.write("**Relationships:**")
                for u, v, data in G.edges(data=True):
                    st.write(f"- {objects[u]['label']} -> {data['relation']} -> {objects[v]['label']}")

if __name__ == "__main__":

    main()
