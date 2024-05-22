import torch
import os
from torch import nn
import numpy as np

# --- Define the Generator Class ---
class Generator(nn.Module):
    def __init__(self, max_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, max_size),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# --- Load Model and Metadata ---
def load_model(path, model_class, default_max_size=256):
    checkpoint = torch.load(path)
    max_size = checkpoint.get('max_size', default_max_size)
    model = model_class(max_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, max_size

def generate_noise(batch_size, noise_dim):
    return torch.randn(batch_size, noise_dim)

def normalize_and_convert_to_hex(data):
    data_min, data_max = data.min(), data.max()
    if data_min == data_max:
        normalized_data = np.zeros_like(data)  # Handle edge case where all values are the same
    else:
        normalized_data = ((data - data_min) / (data_max - data_min) * 255).astype(int)
    
    # Convert the data to hexadecimal format
    hex_data = "".join(f"{x:02x}" for x in normalized_data)
    return hex_data

def reconstruct_level_file(data, metadata, output_path):
    sections = {
        'layer1': "",
        'layer2': "",
        'sprite': "",
        'palette': "",
        'secondary_entrances': "",
        'exanimation': "",
        'exgfx_bypass': ""
    }

    index = 0

    # Assuming metadata is a list of dictionaries and you need the first one
    lengths = metadata[0]['lengths']

    sections['layer1'] = normalize_and_convert_to_hex(data[index:index+lengths['layer1']])
    index += lengths['layer1']

    sections['layer2'] = normalize_and_convert_to_hex(data[index:index+lengths['layer2']])
    index += lengths['layer2']

    sprite_data = data[index:index + lengths['sprite']]
    sections['sprite'] = normalize_and_convert_to_hex(sprite_data)
    index += lengths['sprite']

    sections['palette'] = normalize_and_convert_to_hex(data[index:index+lengths['palette']])
    index += lengths['palette']

    sections['secondary_entrances'] = normalize_and_convert_to_hex(data[index:index+lengths['secondary_entrances']])
    index += lengths['secondary_entrances']

    sections['exanimation'] = normalize_and_convert_to_hex(data[index:index+lengths['exanimation']])
    index += lengths['exanimation']

    sections['exgfx_bypass'] = normalize_and_convert_to_hex(data[index:index+lengths['exgfx_bypass']])
    index += lengths['exgfx_bypass']

    with open(output_path, 'w') as file:
        file.write(f"Version: 832\n")
        file.write(f"Data Pointers Offset: 64, Size: 64\n")
        file.write(f"Special Flags: 00000000\n")
        file.write(f"Banner: Lunar Magic 3.40  \u00A92023 FuSoYa  Defender of Relm\n\n")
        
        file.write(f"Section: Level Information\n")
        file.write(f"Offset: 128, Size: 64\n")
        file.write(f"Data: {'0' * 128}\n\n")

        offset = 192  # Starting offset after Level Information
        for section, hex_data in sections.items():
            size = lengths[section]
            file.write(f"Section: {section.capitalize()} Data\n")
            file.write(f"Offset: {offset}, Size: {size}\n")
            file.write(f"Data: {hex_data}\n\n")
            offset += size

# Load the generator model and metadata
model_path = 'generator_checkpoint.pth'
metadata_path = 'metadata.pth'
netG, max_size = load_model(model_path, Generator)
metadata = torch.load(metadata_path)

# Generate new level data
noise = generate_noise(1, 100)
generated_data = netG(noise).detach().numpy().flatten()

# Reconstruct the level file
output_level_path = 'new_level.txt'
reconstruct_level_file(generated_data, metadata, output_level_path)
