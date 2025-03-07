# # Load the original mscan.py file content
# file_path = "/mnt/data/mscan.py"

# with open(file_path, "r") as file:
#     mscan_code = file.readlines()

# # Identify the line containing the original depthwise convolution definition
# dwconv_line = None
# for i, line in enumerate(mscan_code):
#     if "self.dwconv" in line and "nn.Conv2d" in line:
#         dwconv_line = i
#         break

# # Ensure we found the depthwise convolution line before proceeding
# if dwconv_line is not None:
#     # Create three versions with different shift types
#     shift_types = {
#         "groupshift": "GroupedShiftConv(dim)",
#         "activeshift": "ActiveShiftConv(dim)",
#         "sparseshift": "SparseShiftConv(dim)"
#     }

#     file_paths = {}

#     for shift_name, shift_replacement in shift_types.items():
#         modified_code = mscan_code.copy()

#         # Replace the depthwise convolution line with the shift convolution
#         modified_code[dwconv_line] = f"        self.dwconv = {shift_replacement}\n"

#         # Add import statement at the beginning of the file
#         import_statement = f"from shift_convolutions import {shift_replacement.split('(')[0]}\n"
#         modified_code.insert(0, import_statement)

#         # Save the modified version
#         new_file_path = f"/mnt/data/mscan_{shift_name}.py"
#         with open(new_file_path, "w") as new_file:
#             new_file.writelines(modified_code)

#         file_paths[shift_name] = new_file_path

#     output_message = "Modified files created: mscan_groupshift.py, mscan_activeshift.py, mscan_sparseshift.py."
# else:
#     output_message = "Error: Could not find depthwise convolution definition in mscan.py."

# # Return file paths for user to download
# file_paths
