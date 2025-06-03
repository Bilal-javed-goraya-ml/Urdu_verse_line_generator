import gradio as gr
import torch
import torch.nn as nn
import os
import sys

# Add utils to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from utils.vocab import Vocab
from models.train import Encoder, Decoder , LuongAttention
from utils.load_model import predict_sequence , load_model

# Gradio Interface

def create_interface():
    # Load model on startup
    model_loaded, load_message = load_model()
    
    with gr.Blocks(title="🎭 Urdu Verse line Generator", theme=gr.themes.Soft()) as app:
        gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1>🎭 Urdu Verse line Generator</h1>
                <p>Generate next Verse (line) of Urdu poetry using AI</p>
            </div>
        """)
        
        # Model Status
        with gr.Row():
            gr.HTML(f"""
                <div style="padding: 10px; border-radius: 5px; background-color: {'#d4edda' if model_loaded else '#f8d7da'}; border: 1px solid {'#c3e6cb' if model_loaded else '#f5c6cb'};">
                    <strong>Model Status:</strong> {load_message}
                </div>
            """)
        
        gr.HTML("<hr>")
        
        # Sequence Prediction Tab
        with gr.Tab("🔄 Sequence Prediction"):
            with gr.Row():
                with gr.Column(scale=2):
                    seq_input = gr.Textbox(
                        label="📝 Enter Initial Verse line",
                        placeholder="یہاں اردو شعر کا پہلا مصرع لکھیں...",
                        lines=2,
                        rtl=True
                    )

                    examples = gr.Examples(
                        examples=[
                            ["کالی کالی زلفوں کے پھندے نہ ڈالو"],
                            ["بوتل کھلی ہے رقص میں جامِ شراب ہے"],
                            ["بتا کیا پوچھتا ہے وہ"],
                            ["لاجپال نبی میرے درد دا دوا دینا"],
                        ],
                        inputs=seq_input,
                    )

                    loop_count = gr.Slider(
                        label="🔢 Number of Iterations",
                        minimum=1,
                        maximum=20,
                        value=6,
                        step=1,
                        info="How many times to predict the next Verse line"
                    )
                    seq_btn = gr.Button("🔮 Generate Sequence", variant="primary")
                
                with gr.Column(scale=2):
                    seq_output = gr.Textbox(
                        label="🎭 Generated Sequence",
                        lines=10,
                        rtl=True,
                        interactive=False
                    )
            
            seq_btn.click(
                predict_sequence,
                inputs=[seq_input, loop_count],
                outputs=[seq_output]
            )
        
        # Instructions
        with gr.Tab("📖 Instructions"):
            gr.Markdown("""
                ## How to Use
            
                ### 🔄 Sequence Prediction
                1. Enter an initial Urdu poetry Verse line
                2. Set the number of iterations (how many times to predict)
                3. Click "Generate Sequence" to get a chain of predictions
                4. Each output becomes the input for the next prediction
                
                ### 📝 Tips
                - Make sure the model is trained first: `python main.py --mode train`
                - Use proper Urdu text for better results
                - The model works best with poetry patterns it has seen during training
                
                ### 🔧 Training Commands
                ```bash
                # Build dataset
                python main.py --mode data-builder
                
                # Train model
                python main.py --mode train
                
                # Interactive prediction (CLI)
                python main.py --mode predict
                
                # Run web app
                python app.py
                ```
            """)
        
        gr.HTML("""
            <div style="text-align: center; margin-top: 20px; padding: 10px; color: #666;">
                <small>Built with ❤️ for Urdu Poetry Generation</small>
            </div>
        """)
    
    return app

if __name__ == "__main__":
    # Create and launch the app
    app = create_interface()
    
    print("Starting Urdu Masra Generator Web App...")
    print("The app will open in your browser automatically")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=1102,
        share=True,
        inbrowser=True
    )