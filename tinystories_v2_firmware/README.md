Artifact and full firmware for running the TinyStories_v2 model on NXP S32K3X4EVB-T172 board in the paper "te-comp: A Complete MLIR Compiler Generating Self-Contained C Inference Engines for MCUs"

# Contents

```
tinystories_v2_firmware/
├── code_generation/                     # Run te-comp pipeline and generate self-contained C code for TinyStories_v2 model
├── tinystories_v2.zip                   # The full firmware as S32DS project
└── README.md                            # This file
```

# Prerequisites

- Docker Engine or Docker Desktop
- S32 Design Studio IDE 3.6.10
- RTD 6.0.0, e.g., download SW32K3_S32M27x_RTD_R21-11_6.0.0_D2506_DesignStudio_updatesite.zip
- NXP S32K3X4EVB-T172 board

# Generation Code: Docker (Recommended)

The easiest way to reproduce the TinyStories_v2 self-contained C code is with Docker — it sets up Ubuntu 22.04 with gcc-arm-none-eabi and all Python dependencies, using the pre-built te-comp binary shipped in tools/:

```bash
cd code_generation

# Build the image (< 5 min, no compilation needed)
./docker-build.sh

# Run the te-comp pipeline to generate C code of the model and save results to host
./docker-run.sh --output ./results
```

# Import the Firmware Project and Drop Generated C Files 

To import the project into S32DS:

- Go to the top menu and select File > Import...
- Expand the General folder
- Choose Existing Projects into Workspace and click Next
- Click Browse... and navigate to your .zip file, then open it
- Ensure your project name is checked in the Projects list
- Ensure the Copy projects into workspace option is checked (this prevents modifying the original ZIP content)
- Click Finish. The tinystories_v2 project will now appear in your Project Explorer view

Drop Generated Code to S32DS Project:

```bash
# Go to the generated C files in results folder
cd results

# Copy the header *.h and source *.c files into src folder
```

After that, build and debug/run the project in S32DS. Note that when running, use any serial programs such as minicom or Putty to communicate with the board via UART

