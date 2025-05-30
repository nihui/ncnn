
# must define SRC DST CLASS

file(READ ${SRC} source_data)

# replace
string(TOUPPER ${CLASS} CLASS_UPPER)
string(TOLOWER ${CLASS} CLASS_LOWER)

string(REGEX REPLACE "LAYER_${CLASS_UPPER}_RISCV_H" "LAYER_${CLASS_UPPER}_RISCV_XTHEADVECTOR_H" source_data "${source_data}")
string(REGEX REPLACE "${CLASS}_riscv" "${CLASS}_riscv_xtheadvector" source_data "${source_data}")
string(REGEX REPLACE "#include \"${CLASS_LOWER}_riscv.h\"" "#include \"${CLASS_LOWER}_riscv_xtheadvector.h\"" source_data "${source_data}")

file(WRITE ${DST} "${source_data}")
