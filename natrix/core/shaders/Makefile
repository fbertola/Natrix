
ifndef TARGET
.PHONY: all
all:
	@echo Usage: make TARGET=# [clean, all, rebuild]
	@echo "  TARGET=0 (hlsl  - d3d9)"
	@echo "  TARGET=1 (hlsl  - d3d11)"
	@echo "  TARGET=2 (essl  - nacl)"
	@echo "  TARGET=3 (essl  - android)"
	@echo "  TARGET=4 (glsl)"
	@echo "  TARGET=5 (metal)"
	@echo "  TARGET=6 (pssl)"
	@echo "  TARGET=7 (spriv)"

else

SHADERC=shaderc
SHADERS_DIR=$(addprefix ./, originals/)
ADDITIONAL_INCLUDES?=

ifeq ($(TARGET), 0)
VS_FLAGS=--platform windows -p vs_3_0 -O 3
FS_FLAGS=--platform windows -p ps_3_0 -O 3
SHADER_PATH=compiled/dx9
else
ifeq ($(TARGET), 1)
VS_FLAGS=--platform windows -p vs_5_0 -O 3
FS_FLAGS=--platform windows -p ps_5_0 -O 3
CS_FLAGS=--platform windows -p cs_5_0 -O 1
SHADER_PATH=compiled/dx11
else
ifeq ($(TARGET), 2)
VS_FLAGS=--platform nacl
FS_FLAGS=--platform nacl
SHADER_PATH=compiled/essl
else
ifeq ($(TARGET), 3)
VS_FLAGS=--platform android
FS_FLAGS=--platform android
CS_FLAGS=--platform android
SHADER_PATH=compiled/essl
else
ifeq ($(TARGET), 4)
VS_FLAGS=--platform linux -p 120
FS_FLAGS=--platform linux -p 120
CS_FLAGS=--platform linux -p 430
SHADER_PATH=compiled/glsl
else
ifeq ($(TARGET), 5)
VS_FLAGS=--platform osx -p metal
FS_FLAGS=--platform osx -p metal
CS_FLAGS=--platform osx -p metal
SHADER_PATH=compiled/metal
else
ifeq ($(TARGET), 6)
VS_FLAGS=--platform orbis -p pssl
FS_FLAGS=--platform orbis -p pssl
CS_FLAGS=--platform orbis -p pssl
SHADER_PATH=compiled/pssl
else
ifeq ($(TARGET), 7)
VS_FLAGS=--platform linux -p spirv
FS_FLAGS=--platform linux -p spirv
CS_FLAGS=--platform linux -p spirv
SHADER_PATH=compiled/spirv
endif
endif
endif
endif
endif
endif
endif
endif


CS_FLAGS+=-i <path_to>/bgfx/src/ $(ADDITIONAL_INCLUDES)

BUILD_OUTPUT_DIR=./$(SHADER_PATH)

CS_SOURCES=$(notdir $(wildcard $(addprefix $(SHADERS_DIR), shader.*.comp)))

shader.%.comp:
	@echo $(@)
	$(SILENT) $(SHADERC) $(CS_FLAGS) --type compute -f $(SHADERS_DIR)$(@) -o $(BUILD_OUTPUT_DIR)/$(@)


.PHONY: all
all: dirs $(CS_SOURCES)
	@echo Target $(SHADER_PATH)


.PHONY: dirs
dirs:
	mkdir -p $(BUILD_OUTPUT_DIR)

endif # TARGET
