# ROCm Code Object tooling

ROCm compiler generated code objects (executables, object files, and shared
object libraries) can be examined and code objects extracted with the following
tools.

## roc-obj

High-level wrapper around low-level tooling described below. For a more
detailed overview, see the help text available with `roc-obj --help`.

### Examples:

#### Extract all ROCm code objects from a list of executables
    roc-obj <executable>...

#### Extract all ROCm code objects from a list of executables, and disassemble them
    roc-obj --disassemble <executable>...
    # or
    roc-obj -d <executable>...

#### Extract all ROCm code objects from a list of executables into dir/
    roc-obj --outdir dir/ <executable>...
    # or
    roc-obj -o dir/ <executable>...

#### Extract only ROCm code objects matching regex over Target ID
    roc-obj --target-id gfx9 <executable>...
    # or
    roc-obj -t gfx9 <executable>...

## Low-Level Tooling

### URI syntax:

  ROCm Code Objects can be listed/accessed using the following URI syntax:
```
	code_object_uri ::== file_uri | memory_uri
	file_uri        ::== file:// extract_file [ range_specifier ]
	memory_uri      ::== memory:// process_id range_specifier
	range_specifier ::== [ # | ? ] offset= number & size= number
	extract_file    ::== URI_ENCODED_OS_FILE_PATH
	process_id      ::== DECIMAL_NUMBER
	number          ::== HEX_NUMBER | DECIMAL_NUMBER | OCTAL_NUMBER
```
  Example: file://dir1/dir2/hello_world#offset=133&size=14472
           memory://1234#offset=0x20000&size=3000

### List available ROCm Code Objects: rocm-obj-ls

  Use this tool to list available ROCm code objects.  Code objects are listed by bundle number, entry ID, and URI syntax.

  Usage: roc-obj-ls [-v|h] executable...
  List the URIs of the code objects embedded in the specfied host executables.
    -v Verbose output.  Adds column headers for more human readable format
    -h Show this help message


### Extract ROCm Code Objects: roc-obj-extract

  Extracts available ROCm code objects from specified URI.

  Usage: roc-obj-extract [-o|v|h] URI...
    - URIs can be read from STDIN, one per line.
    - From the URIs specified, extracts code objects into files named: <executable_name>-[pid<number>]-offset<number>-size<number>.co

  Options:
    -o <path> Path for output. If "-" specified, code object is printed to STDOUT.
    -v        Verbose output (includes Entry ID).
    -h        Show this help message

  Note, when specifying a URI argument to roc-obj-extract, if cut and pasting the output from roc-obj-ls you need to escape the '&' character or your shell will interpret it as the option to run the command as a background process.
  As an example, if roc-obj-ls generates a URI like this ```file://my_exe#offset=24576&size=46816xxi```, you need to use the following argument to roc-obj-extract: ```file://my_exe#offset=24576\&size=46816```

### Examples:

#### Dump the ISA for gfx906:
    roc-obj-ls -v <exe> | awk '/gfx906/{print $3}' | roc-obj-extract -o - | llvm-objdump -d - > <exe>.gfx906.isa

#### Check the e_flags of the gfx908 code object:
    roc-obj-ls -v <exe> | awk '/gfx908/{print $3}' | roc-obj-extract -o - | llvm-readelf -h - | grep Flags

#### Disassemble the fourth code object:
    roc-obj-ls <exe> | sed -n 4p | roc-obj-extract -o - | llvm-objdump -d -

#### Sort embedded code objects by size:
    for uri in $(roc-obj-ls <exe>); do printf "%d: %s\n" "$(roc-obj-extract -o - "$uri" | wc -c)" "$uri"; done | sort -n

#### Compare disassembly of gfx803 and gfx900 code objects:
    dis() { roc-obj-ls -v <exe> | grep "$1" | awk '{print $3}' | roc-obj-extract -o - | llvm-objdump -d -; }
    diff <(dis gfx803) <(dis gfx900)

