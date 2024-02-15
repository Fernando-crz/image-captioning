import gradio as gr
import torch
from models import *
from evaluation import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Device being used: ", device)
    # Parâmetros do decoder
    hidden_size = 512
    emb_size = 512
    attention_size = 512
    img_emb_size = 14 * 14
    num_channels = 512
    dropout = 0.4

    # Carregando modelos
    checkpoint = torch.load("checkpoints/best.pth")
    vocab = torch.load("checkpoints/vocab.pth")
    encoder = Encoder().to(device)
    decoder = Decoder(hidden_size, len(vocab), emb_size, attention_size, img_emb_size, num_channels, dropout).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    
    caption_image_interface = gr.Interface(
        fn=lambda image: string_from_caption(caption_image(encoder, decoder, image, vocab)[0]),
        inputs="image",
        outputs="text"
    )

    visualize_attention_interface = gr.Interface(
        fn=lambda image: visualize_attention(encoder, decoder, image, vocab),
        inputs="image",
        outputs="image"
    )

    demo = gr.TabbedInterface(
        [caption_image_interface, visualize_attention_interface],
        ["Caption Image", "Visualize Attention"]
    ).launch(share=False)


if __name__ == "__main__":
    main()