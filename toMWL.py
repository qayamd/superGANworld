import struct

def txt_to_mwl(input_txt_path, output_mwl_path):
    # Read the entire content of the text file
    with open(input_txt_path, 'r') as file:
        lines = file.readlines()

    # Initialize to store section data
    section_data = {}
    version = None
    pointers_offset = None
    pointers_size = None
    special_flags = None
    banner = "Lunar Magic 3.40  ©2023 FuSoYa  Defender of Relm".encode('ascii', 'replace').ljust(48, b'\x00')

    # Parse the lines to extract data
    for line in lines:
        if 'Version:' in line:
            version = int(line.split('Version: ')[1].split('\n')[0].strip())
        elif 'Data Pointers Offset:' in line:
            pointers_offset = int(line.split('Offset: ')[1].split(',')[0].strip())
            pointers_size = int(line.split('Size: ')[1].strip())
        elif 'Special Flags:' in line:
            special_flags = int(line.split('Special Flags: ')[1].strip(), 16)
        elif line.startswith('Section:'):
            section_name = line.split('Section: ')[1].split(' Data')[0].strip()
            section_data[section_name] = {}
        elif 'Offset:' in line:
            section_data[section_name]['offset'] = int(line.split('Offset: ')[1].split(',')[0].strip())
            section_data[section_name]['size'] = int(line.split('Size: ')[1].split('\n')[0].strip())
        elif 'Data:' in line:
            section_data[section_name]['data'] = bytes.fromhex(line.split('Data: ')[1].strip())

    # Write the MWL file using binary format
    with open(output_mwl_path, 'wb') as mwl_file:
        # Write header
        mwl_file.write(b'LM')
        mwl_file.write(struct.pack('<H', version))
        mwl_file.write(struct.pack('<I', pointers_offset))
        mwl_file.write(struct.pack('<I', pointers_size))
        mwl_file.write(struct.pack('<I', special_flags))
        mwl_file.write(banner)

        # Write data pointers
        for section in section_data:
            mwl_file.write(struct.pack('<I', section_data[section]['offset']))
            mwl_file.write(struct.pack('<I', section_data[section]['size']))

        # Write actual data for each section
        for section in section_data:
            mwl_file.write(section_data[section]['data'])

# Convert the generated text file to MWL format
txt_to_mwl('new_level.txt', 'output.mwl')
