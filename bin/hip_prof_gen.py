#!/usr/bin/python
import os, sys, re

verbose = 0
PROF_HEADER = "hip_prof_str.h"
OUTPUT = PROF_HEADER
REC_MAX_LEN = 1024

# Fatal error termination
inp_file = 'none'
line_num = -1
def fatal(msg):
  if line_num != -1:
    print >>sys.stderr, "Error: " + msg + ", file '" + inp_file + "', line (" + str(line_num) + ")"
  else:
    print >>sys.stderr, "Error: " + msg
  sys.exit(1)

# Verbose message
def message(msg):
  if verbose: print >>sys.stdout, msg

#############################################################
# Normalizing API arguments
def filtr_api_args(args_str):
  args_str = re.sub(r'^\s*', r'', args_str);
  args_str = re.sub(r'\s*$', r'', args_str);
  args_str = re.sub(r'\s*,\s*', r',', args_str);
  args_str = re.sub(r'\s+', r' ', args_str);
  args_str = re.sub(r'void \*', r'void* ', args_str);
  args_str = re.sub(r'(enum|struct) ', '', args_str);
  return args_str

# Normalizing types
def norm_api_types(type_str):
  type_str = re.sub(r'uint32_t', r'unsigned int', type_str)
  type_str = re.sub(r'^unsigned$', r'unsigned int', type_str)
  return type_str

# Creating a list of arguments [(type, name), ...]
def list_api_args(args_str):
  args_str = filtr_api_args(args_str)
  args_list = []
  if args_str != '':
    for arg_pair in args_str.split(','):
      if arg_pair == 'void': continue
      arg_pair = re.sub(r'\s*=\s*\S+$','', arg_pair);
      m = re.match("^(.*)\s(\S+)$", arg_pair);
      if m:
        arg_type = norm_api_types(m.group(1))
        arg_name = m.group(2)
        args_list.append((arg_type, arg_name))
      else:
        fatal("bad args: args_str: '" + args_str + "' arg_pair: '" + arg_pair + "'")
  return args_list;

# Creating arguments string "type0, type1, ..."
def filtr_api_types(args_str):
  args_list = list_api_args(args_str)
  types_str = ''
  for arg_tuple in args_list:
    types_str += arg_tuple[0] + ', '
  return types_str

# Creating options list [opt0, opt1, ...]
def filtr_api_opts(args_str):
  args_list = list_api_args(args_str)
  opts_list = []
  for arg_tuple in args_list:
    opts_list.append(arg_tuple[1])
  return opts_list
#############################################################
# Parsing API header
# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
def parse_api(inp_file_p, out):
  global inp_file
  global line_num
  inp_file = inp_file_p

  beg_pattern = re.compile("^(hipError_t|const char\s*\*)\s+[^\(]+\(");
  api_pattern = re.compile("^(hipError_t|const char\s*\*)\s+([^\(]+)\(([^\)]*)\)");
  end_pattern = re.compile("Texture");
  hidden_pattern = re.compile(r'__attribute__\(\(visibility\("hidden"\)\)\)')
  nms_open_pattern = re.compile(r'namespace hip_impl {')
  nms_close_pattern = re.compile(r'}')

  inp = open(inp_file, 'r')

  found = 0
  hidden = 0
  nms_level = 0;
  record = ""
  line_num = -1 

  for line in inp.readlines():
    record += re.sub(r'^\s+', r' ', line[:-1])
    line_num += 1

    if len(record) > REC_MAX_LEN:
      fatal("bad record \"" + record + "\"")

    if beg_pattern.match(record) and (hidden == 0) and (nms_level == 0): found = 1

    if found != 0:
      record = re.sub("\s__dparm\([^\)]*\)", '', record);
      m = api_pattern.match(record)
      if m:
        found = 0
        if end_pattern.search(record): break
        out[m.group(2)] = m.group(3)
      else: continue

    hidden = 0
    if hidden_pattern.match(line): hidden = 1

    if nms_open_pattern.match(line): nms_level += 1
    if (nms_level > 0) and nms_close_pattern.match(line): nms_level -= 1
    if nms_level < 0:
      fatal("nms level < 0")

    record = ""

  inp.close()
  line_num = -1
#############################################################
# Patching API implementation
# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
#    HIP_INIT_CB(hipSetupArgument, arg, size, offset);
# inp_file - input implementation source file
# api_map - input public API map [<api name>] => <api args>
# out - output map  [<api name>] => [opt0, opt1, ...]
def parse_content(inp_file_p, api_map, out):
  global inp_file
  global line_num
  inp_file = inp_file_p

  # API definition begin pattern
  beg_pattern = re.compile("^(hipError_t|const char\s*\*)\s+[^\(]+\(");
  # API definition complete pattern
  api_pattern = re.compile("^(hipError_t|const char\s*\*)\s+([^\(]+)\(([^\)]*)\)\s*{");
  # API init macro pattern
  init_pattern = re.compile("^\s*HIP_INIT[_\w]*_API\(([^,]+)(,|\))");
  target_pattern = re.compile("^(\s*HIP_INIT[^\(]*)(_API\()(.*)\);\s*$");

  # Open input file
  inp = open(inp_file, 'r')

  # API name
  api_name = ""
  # Valid public API found flag
  api_valid = 0

  # Input file patched content
  content = ''
  # Sub content for found API defiition
  sub_content = ''
  # Current record, accumulating several API definition related lines
  record = ''
  # Current input file line number
  line_num = -1 
  # API beginning found flag
  found = 0

  # Reading input file
  for line in inp.readlines():
    # Accumulating record
    record += re.sub(r'^\s+', r' ', line[:-1])
    line_num += 1

    if len(record) > REC_MAX_LEN:
      fatal("bad record \"" + record + "\"")
      break;

    # Looking for API begin
    if beg_pattern.match(record): found = 1

    # Matching complete API definition
    if found == 1:
      record = re.sub("\s__dparm\([^\)]*\)", '', record);
      m = api_pattern.match(record)
      # Checking if complete API matched
      if m:
        found = 2
        api_name = m.group(2);
        # Checking if API name is in the API map
        if api_name in api_map:
          # Getting API arguments
          api_args = m.group(3)
          # Getting etalon arguments from the API map
          eta_args = api_map[api_name]
          if eta_args == '':
            eta_args = api_args
            api_map[api_name] = eta_args
          # Normalizing API arguments
          api_types = filtr_api_types(api_args)
          # Normalizing etalon arguments
          eta_types = filtr_api_types(eta_args)
          if api_types == eta_types:
            # API is already found
            if api_name in out:
              fatal("API redefined \"" + api_name + "\", record \"" + record + "\"")
            # Set valid public API found flag
            api_valid = 1
            # Set output API map with API arguments list
            out[api_name] = filtr_api_opts(api_args)
          else:
            # Warning about mismatched API, possible non public overloaded version
            api_diff = '\t\t' + inp_file + " line(" + str(line_num) + ")\n\t\tapi: " + api_types + "\n\t\teta: " + eta_types
            message("\t" + api_name + ':\n' + api_diff + '\n')

    # API found action
    if found == 2:
      # Looking for INIT macro
      m = init_pattern.match(line)
      if m:
        found = 0
        if api_valid == 1:
          api_valid = 0
          message("\t" + api_name)
        else:
          # Registering dummy API for non public API if the name in INIT is not NONE
          init_name = m.group(1)
          # Ignore if it is initialized as NONE
          if init_name != 'NONE':
            # Check if init name matching API name
            if init_name != api_name:
              fatal("init name mismatch: '" + init_name +  "' <> '" + api_name + "'")
            # If init name is not in public API map then it is private API
            # else it was not identified and will be checked on finish
            if not init_name in api_map:
              if init_name in out:
                fatal("API reinit \"" + api_name + "\", record \"" + record + "\"")
              out[init_name] = []
      elif re.search('}', line):
        found = 0
        # Expect INIT macro for valid public API
        if api_valid == 1:
          api_valid = 0
          if api_name in out:
            del out[api_name]
            del api_map[api_name]
            out['.' + api_name] = 1
          else:
            fatal("API is not in out \"" + api_name + "\", record \"" + record + "\"")

    if found != 1: record = ""
    content += line

  inp.close()
  line_num = -1

  if len(out) != 0:
    return content
  else:
    return '' 

# src path walk
def parse_src(api_map, src_path, src_patt, out):
  pattern = re.compile(src_patt)
  src_path = re.sub(r'\s', '', src_path)
  for src_dir in src_path.split(':'):
    message("Parsing " + src_dir + " for '" + src_patt + "'")
    for root, dirs, files in os.walk(src_dir):
      for fnm in files:
        if pattern.search(fnm):
          file = root + '/' + fnm
          message(file)
          content = parse_content(file, api_map, out);
          if content != '':
            f = open(file, 'w')
            f.write(content)
            f.close()
#############################################################
# Generating profiling primitives header
# api_map - public API map [<api name>] => [(type, name), ...]
# opts_map - opts map  [<api name>] => [opt0, opt1, ...]
def generate_prof_header(f, api_map, opts_map):
  # Private API list
  priv_lst = []

  f.write('// automatically generated sources\n')
  f.write('#ifndef _HIP_PROF_STR_H\n');
  f.write('#define _HIP_PROF_STR_H\n');
  f.write('#include <sstream>\n');
  f.write('#include <string>\n');
  
  # Generating dummy macro for non-public API
  f.write('\n// Dummy API primitives\n')
  f.write('#define INIT_NONE_CB_ARGS_DATA(cb_data) {};\n')
  for name in opts_map:
    if not name in api_map:
      opts_lst = opts_map[name]
      if len(opts_lst) != 0:
        fatal("bad dummy API \"" + name + "\", args: " + str(opts_lst))
      f.write('#define INIT_'+ name + '_CB_ARGS_DATA(cb_data) {};\n')
      priv_lst.append(name)
  
  for name in priv_lst:
    message("Private: " + name)
  
  # Generating the callbacks ID enumaration
  f.write('\n// HIP API callbacks ID enumaration\n')
  f.write('enum hip_api_id_t {\n')
  cb_id = 0
  for name in api_map.keys():
    f.write('  HIP_API_ID_' + name + ' = ' + str(cb_id) + ',\n')
    cb_id += 1
  f.write('  HIP_API_ID_NUMBER = ' + str(cb_id) + ',\n')
  f.write('  HIP_API_ID_ANY = ' + str(cb_id + 1) + ',\n')
  f.write('\n')
  f.write('  HIP_API_ID_NONE = HIP_API_ID_NUMBER,\n')
  for name in priv_lst:
    f.write('  HIP_API_ID_' + name + ' = HIP_API_ID_NUMBER,\n')
  f.write('};\n')
  
  # Generating the callbacks ID enumaration
  f.write('\n// Return HIP API string\n')
  f.write('static const char* hip_api_name(const uint32_t& id) {\n')
  f.write('  switch(id) {\n')
  for name in api_map.keys():
    f.write('    case HIP_API_ID_' + name + ': return "' +  name + '";\n')
  f.write('  };\n')
  f.write('  return "unknown";\n')
  f.write('};\n')
  
  # Generating the callbacks data structure
  f.write('\n// HIP API callbacks data structure\n')
  f.write(
  'struct hip_api_data_t {\n' +
  '  uint64_t correlation_id;\n' +
  '  uint32_t phase;\n' +
  '  union {\n'
  )
  for name, args in api_map.items():
    if len(args) != 0:
      f.write('    struct {\n')
      for arg_tuple in args:
        f.write('      ' + arg_tuple[0] + ' ' + arg_tuple[1] + ';\n')
      f.write('    } ' + name + ';\n')
  f.write(
  '  } args;\n' +
  '};\n'
  )
  
  # Generating the callbacks args data filling macros
  f.write('\n// HIP API callbacks args data filling macros\n')
  for name, args in api_map.items():
    f.write('// ' + name + str(args) + '\n')
    f.write('#define INIT_' + name + '_CB_ARGS_DATA(cb_data) { \\\n')
    if name in opts_map:
      opts_list = opts_map[name]
      if len(args) != len(opts_list):
        fatal("\"" + name + "\" API args and opts mismatch, args: " + str(args) + ", opts: " + str(opts_list))
      # API args iterating:
      #   type is args[<ind>][0]
      #   name is args[<ind>][1]
      for ind in range(0, len(args)):
        arg_tuple = args[ind]
        fld_name = arg_tuple[1]
        arg_name = opts_list[ind]
        f.write('  cb_data.args.' + name + '.' + fld_name + ' = ' + arg_name + '; \\\n')
    f.write('};\n')
  f.write('#define INIT_CB_ARGS_DATA(cb_id, cb_data) INIT_##cb_id##_CB_ARGS_DATA(cb_data)\n')
  
  # Generating the method for the API string, name and parameters
  f.write('\n')
  f.write('#if 0\n')
  f.write('// HIP API string method, method name and parameters\n')
  f.write('const char* hipApiString(hip_api_id_t id, const hip_api_data_t* data) {\n')
  f.write('  std::ostringstream oss;\n')
  f.write('  switch (id) {\n')
  for name, args in api_map.items():
    f.write('    case HIP_API_ID_' + name + ':\n')
    f.write('      oss << "' + name + '("')
    for ind in range(0, len(args)):
      arg_tuple = args[ind]
      arg_name = arg_tuple[1]
      if ind != 0: f.write(' << ","')
      f.write('\n          << " ' + arg_name  + '=" << data->args.' + name + '.' + arg_name)
    f.write('\n          << ")";\n')
    f.write('    break;\n')
  f.write('    default: oss << "unknown";\n')
  f.write('  };\n')
  f.write('  return strdup(oss.str().c_str());\n')
  f.write('};\n')
  f.write('#endif\n')
  
  f.write('#endif  // _HIP_PROF_STR_H\n');

#############################################################
# main
# Usage
if (len(sys.argv) > 1) and (sys.argv[1] == '-v'):
  verbose = 1
  sys.argv.pop(1)

if (len(sys.argv) < 3):
  fatal ("Usage: " + sys.argv[0] + " [-v] <input HIP API .h file> <patched srcs path>\n" +
         "  -v - verbose messages\n" +
         "  example:\n" +
         "  $ hipap.py hip/include/hip/hcc_detail/hip_runtime_api.h hip/src")

# API header file given as an argument
api_hfile = sys.argv[1]
if not os.path.isfile(api_hfile):
  fatal("input file '" + api_hfile + "' not found")

# Srcs directory given as an argument
src_pat = "\.cpp$"
src_dir = sys.argv[2]
if not os.path.isdir(src_dir):
  fatal("src directory " + src_dir + "' not found")

if len(sys.argv) > 3: OUTPUT = sys.argv[3]

# API declaration map
api_map = {
  'hipHccModuleLaunchKernel': ''
}
# API options map
opts_map = {}

# Parsing API header
parse_api(api_hfile, api_map)

# Parsing sources
parse_src(api_map, src_dir, src_pat, opts_map)

# Checking for non-conformant APIs
for name in opts_map.keys():
  m = re.match(r'\.(\S*)', name)
  if m:
    message("Init missing: " + m.group(1))
    del opts_map[name]

# Converting api map to map of lists
# Checking for not found APIs
not_found = 0
if len(opts_map) != 0:
  for name in api_map.keys():
    args_str = api_map[name];
    api_map[name] = list_api_args(args_str)
    if not name in opts_map:
      fatal("not found: " + name)
      not_found += 1
if not_found != 0:
  fatal(not_found + " API calls not found")

# Generating output header file
with open(OUTPUT, 'w') as f:
  generate_prof_header(f, api_map, opts_map)

# Successfull exit
sys.exit(0)
