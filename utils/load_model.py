import os
import sys
import torch
# Add utils to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from utils.vocab import Vocab
from models.train import Encoder, Decoder , LuongAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Variables
encoder = None
decoder = None
vocab = None

def load_model():
    """Load the trained model"""
    global encoder, decoder, vocab
    
    if not os.path.exists("trained/model.pt"):
        return False, "❌ No trained model found. Please train the model first using: python main.py --mode train"
    
    try:
        # Load vocabulary
        vocab = Vocab()
        
        # Load model
        checkpoint = torch.load("trained/model.pt", map_location=device)
    
        # vocab_size = len(vocab)
        # emb_size = 512  # Default from training
        # hidden_size = 512  # Default from training
        vocab_size = checkpoint["vocab_size"]
        emb_size = checkpoint["emb_size"]
        hidden_size = checkpoint["hidden_size"]
        
        encoder = Encoder(vocab_size, emb_size, hidden_size).to(device)
        decoder = Decoder(vocab_size, emb_size, hidden_size).to(device)
        
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        
        encoder.eval()
        decoder.eval()
        
        return True, "✅ Model loaded successfully!"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"


# def predict_sequence(input_text, loop_count, max_length=50):
#     """Predict sequence of masras with loop"""
#     global encoder, decoder, vocab
    
#     if encoder is None or decoder is None or vocab is None:
#         return "❌ Model not loaded. Please load the model first."
    
#     if not input_text.strip():
#         return "❌ Please enter a masra."
    
#     if loop_count < 1:
#         return "❌ Loop count must be at least 1."
    
#     try:
#         results = []
#         current_input = input_text.strip()
#         results.append(f"{current_input}")
        
#         for i in range(loop_count):
#             # Predict next masra
#             with torch.no_grad():
#                 input_seq = vocab.encode(current_input)
#                 src = torch.tensor(input_seq).unsqueeze(0).to(device)
                
#                 encoder_outputs, hidden, cell = encoder(src)
#                 input_token = torch.tensor([vocab.sos_idx]).to(device)
#                 decoded_words = []
                
#                 for _ in range(max_length):
#                     output, hidden, cell = decoder(input_token, hidden, cell, encoder_outputs)
#                     predicted_id = output.argmax(dim=-1).item()
                    
#                     if predicted_id == vocab.eos_idx:
#                         break
                        
#                     if predicted_id not in {vocab.pad_idx, vocab.sos_idx, vocab.unk_idx}:
#                         decoded_words.append(vocab.idx2word[predicted_id])
                    
#                     input_token = torch.tensor([predicted_id]).to(device)
                
#                 predicted_masra = " ".join(decoded_words) if decoded_words else "[Could not generate]"
#                 results.append(f"{predicted_masra}")
                
#                 # Use the predicted masra as input for next iteration
#                 current_input = predicted_masra
                
#                 # Stop if we couldn't generate anything meaningful
#                 if predicted_masra == "[Could not generate]":
#                     break
        
#         return "\n".join(results)
    
#     except Exception as e:
#         return f"❌ Error during sequence prediction: {str(e)}"


def predict_sequence(input_text, max_length=50):
    """Predict sequence of masras until 'ختم' is predicted"""
    global encoder, decoder, vocab

    if encoder is None or decoder is None or vocab is None:
        return "❌ Model not loaded. Please load the model first."

    if not input_text.strip():
        return "❌ Please enter a masra."

    try:
        results = []
        current_input = input_text.strip()
        results.append(current_input)

        for _ in range(50):  # Max limit to avoid infinite loop
            with torch.no_grad():
                input_seq = vocab.encode(current_input)
                src = torch.tensor(input_seq).unsqueeze(0).to(device)

                encoder_outputs, hidden, cell = encoder(src)
                input_token = torch.tensor([vocab.sos_idx]).to(device)
                decoded_words = []

                for _ in range(max_length):
                    output, hidden, cell = decoder(input_token, hidden, cell, encoder_outputs)
                    predicted_id = output.argmax(dim=-1).item()

                    if predicted_id == vocab.eos_idx:
                        break

                    if predicted_id not in {vocab.pad_idx, vocab.sos_idx, vocab.unk_idx}:
                        decoded_words.append(vocab.idx2word[predicted_id])

                    input_token = torch.tensor([predicted_id]).to(device)

                predicted_masra = " ".join(decoded_words) if decoded_words else "[Could not generate]"

                # 🚫 Stop if predicted masra is "ختم"
                if predicted_masra.strip() == "ختم":
                    break

                results.append(predicted_masra)
                current_input = predicted_masra

                if predicted_masra == "[Could not generate]":
                    break

        return "\n".join(results)

    except Exception as e:
        return f"❌ Error during sequence prediction: {str(e)}"
