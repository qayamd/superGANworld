import struct
import os

def parse_mwl_header(file):
    header = file.read(64)
    version = struct.unpack('<H', header[2:4])[0]
    offset_data_pointers = struct.unpack('<I', header[4:8])[0]
    size_data_pointers = struct.unpack('<I', header[8:12])[0]
    special_flags = header[12:16]
    banner_string = header[16:].decode('utf-8', errors='replace').strip()
    return version, offset_data_pointers, size_data_pointers, special_flags, banner_string

def read_data_pointers(file, offset, count):
    file.seek(offset)
    pointers = []
    for _ in range(count):
        start, size = struct.unpack('<II', file.read(8))
        pointers.append((start, size))
    return pointers

def read_section_data(file, pointers):
    sections = {}
    section_names = ["Level Information", "Layer 1 Data", "Layer 2 Data", "Sprite Data", "Palette Data",
                     "Secondary Entrances", "ExAnimation Data", "ExGFX and Bypass Information"]
    for name, (offset, size) in zip(section_names, pointers):
        file.seek(offset)
        data = file.read(size)
        sections[name] = data
    return sections

def save_to_text_file(filepath, header, pointers, sections):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Version: {header[0]}\n")
        f.write(f"Data Pointers Offset: {header[1]}, Size: {header[2]}\n")
        f.write(f"Special Flags: {header[3].hex()}\n")
        f.write(f"Banner: {header[4]}\n\n")

        for name, (offset, size) in zip(sections.keys(), pointers):
            f.write(f"Section: {name}\n")
            f.write(f"Offset: {offset}, Size: {size}\n")
            f.write(f"Data: {sections[name].hex()}\n\n")

def convert_mwl_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith('.mwl'):
            mwl_path = os.path.join(input_folder, filename)
            text_path = os.path.join(output_folder, filename.replace('.mwl', '.txt'))

            with open(mwl_path, 'rb') as file:
                version, offset_data_pointers, size_data_pointers, flags, banner = parse_mwl_header(file)
                #TODO: Account for weird custom levels so the model doesnt hallucinate spare sections of data
                pointers = read_data_pointers(file, offset_data_pointers, 8)  #The level should alwys have 8 sections if it is valid
                sections = read_section_data(file, pointers)
                save_to_text_file(text_path, (version, offset_data_pointers, size_data_pointers, flags, banner), pointers, sections)


input_folder_path = r'./MWLs'
output_folder_path =r"./Data"

convert_mwl_files(input_folder_path, output_folder_path)
