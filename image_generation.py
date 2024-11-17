import torch
import matplotlib.pyplot as plt
from anime_face_generation_GAN import Generator  # Adjust the import path if needed

def generate_single_image(model_path, latent_size, device):
    # Load the trained model
    generator = Generator(latent_size).to(device)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    # Generate a random latent vector
    latent_vector = torch.randn(1, latent_size, 1, 1, device=device)

    # Generate the image
    with torch.no_grad():
        generated_image = generator(latent_vector)

    # Process and visualize the generated image
    generated_image = generated_image.squeeze(0)  # Remove the batch dimension
    generated_image = generated_image.permute(1, 2, 0)  # Change the shape to (H, W, C)
    generated_image = (generated_image.cpu().numpy() + 1) / 2  # Convert to numpy array and scale to [0, 1]

    plt.imshow(generated_image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Set the path to your model file
    model_path = "generator-3.pth"
    latent_size = 256
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    generate_single_image(model_path, latent_size, device)
