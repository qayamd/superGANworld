import os
import re
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import pad
import torch.optim as optim

# Initialize max_size, dynamically updated later
max_size = 256

# --- Data Parsing and Preprocessing ---
def hex_to_int_list(hex_data):
    return [int(hex_data[i:i+2], 16) for i in range(0, len(hex_data), 2)]

def parse_level_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    patterns = {
        'layer1': r'Section: Layer 1 Data\nOffset: \d+, Size: \d+\nData: ([0-9a-f]+)',
        'layer2': r'Section: Layer 2 Data\nOffset: \d+, Size: \d+\nData: ([0-9a-f]+)',
        'sprite': r'Section: Sprite Data\nOffset: \d+, Size: \d+\nData: ([0-9a-f]+)',
        'palette': r'Section: Palette Data\nOffset: \d+, Size: \d+\nData: ([0-9a-f]+)',
        'secondary_entrances': r'Section: Secondary Entrances\nOffset: \d+, Size: \d+\nData: ([0-9a-f]+)',
        'exanimation': r'Section: ExAnimation Data\nOffset: \d+, Size: \d+\nData: ([0-9a-f]+)',
        'exgfx_bypass': r'Section: ExGFX and Bypass Information\nOffset: \d+, Size: \d+\nData: ([0-9a-f]+)'
    }

    data = {key: [] for key in patterns.keys()}
    lengths = {key: 0 for key in patterns.keys()}

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            data[key] = hex_to_int_list(match.group(1))
            lengths[key] = len(data[key])

    sprite_objects = []
    for i in range(0, len(data['sprite']) - len(data['sprite']) % 3, 3):
        sprite_objects.append({'id': data['sprite'][i], 'x': data['sprite'][i+1], 'y': data['sprite'][i+2]})
    data['sprite'] = sprite_objects

    combined_data = data['layer1'] + data['layer2'] + [item for sublist in sprite_objects for item in sublist.values()] + data['palette'] + data['secondary_entrances'] + data['exanimation'] + data['exgfx_bypass']
    data['combined'] = combined_data

    metadata = {
        'lengths': lengths,
        'total_length': len(combined_data)
    }

    return data, metadata

def process_directory(directory_path):
    global max_size
    all_data = []
    all_metadata = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            level_data, metadata = parse_level_file(file_path)
            all_data.append(level_data)
            all_metadata.append(metadata)
            current_size = len(level_data['combined'])
            if current_size > max_size:
                max_size = current_size

    for data in all_data:
        current_combined = torch.tensor(data['combined'], dtype=torch.float32)
        if current_combined.numel() < max_size:
            padded_combined = pad(current_combined, (0, max_size - current_combined.numel()), 'constant', 0)
        else:
            padded_combined = current_combined
        data['combined'] = padded_combined

    return all_data, all_metadata

# --- Neural Network Definitions ---
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

class Discriminator(nn.Module):
    def __init__(self, max_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(max_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

# --- Model Save and Load Functions ---
def save_model(model, path, max_size):
    torch.save({
        'model_state_dict': model.state_dict(),
        'max_size': max_size
    }, path)

def load_model(path, model_class, default_max_size=256):
    checkpoint = torch.load(path)
    max_size = checkpoint.get('max_size', default_max_size)
    model = model_class(max_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# --- Data Preparation ---
directory_path = r"./Data"
processed_data, metadata = process_directory(directory_path)
combined_data = torch.stack([data['combined'] for data in processed_data])
metadata_tensor = torch.tensor([list(meta['lengths'].values()) + [meta['total_length']] for meta in metadata])
dataset = TensorDataset(combined_data, metadata_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- Training Setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator(max_size).to(device)
netD = Discriminator(max_size).to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
num_epochs = 500
real_label = 1
fake_label = 0

# --- Training Loop ---
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Update Discriminator
        netD.zero_grad()
        real_cpu, _ = data
        real_cpu = real_cpu.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Generate fake data
        noise = torch.randn(batch_size, 100, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update Generator
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # Optionally save checkpoints
    if epoch % 100 == 0:
        save_model(netG, 'generator_checkpoint.pth', max_size)
        save_model(netD, 'discriminator_checkpoint.pth', max_size)

# Save metadata
torch.save(metadata, 'metadata.pth')
