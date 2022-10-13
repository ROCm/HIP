#!/usr/bin/env perl
# Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Need perl > 5.10 to use logic-defined or
use 5.006; use v5.10.1;
use warnings;
use File::Basename;
use File::Temp qw/ :mktemp  /;
use Cwd;
use Cwd 'abs_path';

# HIP compiler driver
# Will call clang or nvcc (depending on target) and pass the appropriate include and library options for
# the target compiler and HIP infrastructure.

# Will pass-through options to the target compiler.  The tools calling HIPCC must ensure the compiler
# options are appropriate for the target compiler.

# Environment variable HIP_PLATFORM is to detect amd/nvidia path:
# HIP_PLATFORM='nvidia' or HIP_PLATFORM='amd'.
# If HIP_PLATFORM is not set hipcc will attempt auto-detect based on if nvcc is found.
#
# Other environment variable controls:
# HIP_PATH       : Path to HIP directory, default is one dir level above location of this script.
# CUDA_PATH      : Path to CUDA SDK (default /usr/local/cuda). Used on NVIDIA platforms only.
# HSA_PATH       : Path to HSA dir (defaults to ../../hsa relative to abs_path
#                  of this script). Used on AMD platforms only.
# HIP_ROCCLR_HOME : Path to HIP/ROCclr directory. Used on AMD platforms only.
# HIP_CLANG_PATH : Path to HIP-Clang (default to ../../llvm/bin relative to this
#                  script's abs_path). Used on AMD platforms only.

if(scalar @ARGV == 0){
    print "No Arguments passed, exiting ...\n";
    exit(-1);
}

# retrieve --rocm-path hipcc option from command line.
# We need to respect this over the env var ROCM_PATH for this compilation.
sub get_rocm_path_option {
  my $rocm_path="";
  my @CLArgs = @ARGV;
  foreach $arg (@CLArgs) {
    if (index($arg,"--rocm-path=") != -1) {
      ($rocm_path) = $arg=~ /=\s*(.*)\s*$/;
    }
  }
  return $rocm_path;
}

$verbose = $ENV{'HIPCC_VERBOSE'} // 0;
# Verbose: 0x1=commands, 0x2=paths, 0x4=hipcc args

$HIPCC_COMPILE_FLAGS_APPEND=$ENV{'HIPCC_COMPILE_FLAGS_APPEND'};
$HIPCC_LINK_FLAGS_APPEND=$ENV{'HIPCC_LINK_FLAGS_APPEND'};

# Known Features
@knownFeatures = ('sramecc-', 'sramecc+', 'xnack-', 'xnack+');

$HIP_LIB_PATH=$ENV{'HIP_LIB_PATH'};
$DEVICE_LIB_PATH=$ENV{'DEVICE_LIB_PATH'};
$HIP_CLANG_HCC_COMPAT_MODE=$ENV{'HIP_CLANG_HCC_COMPAT_MODE'}; # HCC compatibility mode
$HIP_COMPILE_CXX_AS_HIP=$ENV{'HIP_COMPILE_CXX_AS_HIP'} // "1";

#---
# Temporary directories
my @tmpDirs = ();

#---
# Create a new temporary directory and return it
sub get_temp_dir {
    my $tmpdir = mkdtemp("/tmp/hipccXXXXXXXX");
    push (@tmpDirs, $tmpdir);
    return $tmpdir;
}

#---
# Delete all created temporary directories
sub delete_temp_dirs {
    if (@tmpDirs) {
        system ('rm -rf ' . join (' ', @tmpDirs));
    }
    return 0;
}

my $base_dir;
my $rocmPath;
BEGIN {
    $base_dir = dirname(Cwd::realpath(__FILE__) );
    $rocmPath = get_rocm_path_option();
    if ($rocmPath ne '') {
      # --rocm-path takes precedence over ENV{ROCM_PATH}
      $ENV{ROCM_PATH}=$rocmPath;
    }
}
use lib "$base_dir/";

use hipvars;
$isWindows      =   $hipvars::isWindows;
$HIP_RUNTIME    =   $hipvars::HIP_RUNTIME;
$HIP_PLATFORM   =   $hipvars::HIP_PLATFORM;
$HIP_COMPILER   =   $hipvars::HIP_COMPILER;
$HIP_CLANG_PATH =   $hipvars::HIP_CLANG_PATH;
$CUDA_PATH      =   $hipvars::CUDA_PATH;
$HIP_PATH       =   $hipvars::HIP_PATH;
$ROCM_PATH      =   $hipvars::ROCM_PATH;
$HIP_VERSION    =   $hipvars::HIP_VERSION;
$HSA_PATH       =   $hipvars::HSA_PATH;
$HIP_ROCCLR_HOME =   $hipvars::HIP_ROCCLR_HOME;

if ($HIP_PLATFORM eq "amd") {
  # If using ROCclr runtime, need to find HIP_ROCCLR_HOME
  if (!defined $DEVICE_LIB_PATH and -e "$HIP_ROCCLR_HOME/lib/bitcode") {
    $DEVICE_LIB_PATH = "$HIP_ROCCLR_HOME/lib/bitcode";
  }
  $HIP_INCLUDE_PATH = "$HIP_ROCCLR_HOME/include";
  if (!defined $HIP_LIB_PATH) {
    $HIP_LIB_PATH = "$HIP_ROCCLR_HOME/lib";
  }

  if (!defined $DEVICE_LIB_PATH) {
    if (-e "$ROCM_PATH/amdgcn/bitcode") {
      $DEVICE_LIB_PATH = "$ROCM_PATH/amdgcn/bitcode";
    }
    else {
      # This path is to support an older build of the device library
      # TODO: To be removed in the future.
      $DEVICE_LIB_PATH = "$ROCM_PATH/lib";
    }
  }
}

if ($verbose & 0x2) {
    print ("HIP_PATH=$HIP_PATH\n");
    print ("HIP_PLATFORM=$HIP_PLATFORM\n");
    print ("HIP_COMPILER=$HIP_COMPILER\n");
    print ("HIP_RUNTIME=$HIP_RUNTIME\n");
}

# set if user explicitly requests -stdlib=libc++. (else we default to libstdc++ for better interop with g++):
$setStdLib = 0;  # TODO - set to 0

$default_amdgpu_target = 1;

if ($HIP_PLATFORM eq "amd") {
    $execExtension = "";
    if($isWindows) {
        $execExtension = ".exe";
    } 
    $HIPCC="$HIP_CLANG_PATH/clang++" . $execExtension;

    # If $HIPCC clang++ is not compiled, use clang instead
    if ( ! -e $HIPCC ) {
        $HIPCC="$HIP_CLANG_PATH/clang" . $execExtension;
        $HIPLDFLAGS = "--driver-mode=g++";
    }
    # to avoid using dk linker or MSVC linker
    if($isWindows) {
        $HIPLDFLAGS .= " -fuse-ld=lld";
        $HIPLDFLAGS .= " --ld-path=$HIP_CLANG_PATH/lld-link.exe";
    }
    $HIP_CLANG_VERSION = `$HIPCC --version`;
    $HIP_CLANG_VERSION=~/.*clang version (\S+).*/;
    $HIP_CLANG_VERSION=$1;

    # Figure out the target with which llvm is configured
    $HIP_CLANG_TARGET = `$HIPCC -print-target-triple`;
    chomp($HIP_CLANG_TARGET);

    if (! defined $HIP_CLANG_INCLUDE_PATH) {
        $HIP_CLANG_INCLUDE_PATH = abs_path("$HIP_CLANG_PATH/../lib/clang/$HIP_CLANG_VERSION/include");
    }
    if (! defined $HIP_INCLUDE_PATH) {
        $HIP_INCLUDE_PATH = "$HIP_PATH/include";
    }
    if (! defined $HIP_LIB_PATH) {
        $HIP_LIB_PATH = "$HIP_PATH/lib";
    }
    if ($verbose & 0x2) {
        print ("ROCM_PATH=$ROCM_PATH\n");
        if (defined $HIP_ROCCLR_HOME) {
            print ("HIP_ROCCLR_HOME=$HIP_ROCCLR_HOME\n");
        }
        print ("HIP_CLANG_PATH=$HIP_CLANG_PATH\n");
        print ("HIP_CLANG_INCLUDE_PATH=$HIP_CLANG_INCLUDE_PATH\n");
        print ("HIP_INCLUDE_PATH=$HIP_INCLUDE_PATH\n");
        print ("HIP_LIB_PATH=$HIP_LIB_PATH\n");
        print ("DEVICE_LIB_PATH=$DEVICE_LIB_PATH\n");
        print ("HIP_CLANG_TARGET=$HIP_CLANG_TARGET\n");
    }

    $HIPCXXFLAGS .= " -isystem \"$HIP_CLANG_INCLUDE_PATH/..\"";
    $HIPCFLAGS .= " -isystem \"$HIP_CLANG_INCLUDE_PATH/..\"";
    $HIPLDFLAGS .= " -L\"$HIP_LIB_PATH\"";
    if ($isWindows) {
      $HIPLDFLAGS .= " -lamdhip64";
    }
    if ($HIP_CLANG_HCC_COMPAT_MODE) {
        ## Allow __fp16 as function parameter and return type.
        $HIPCXXFLAGS .= " -Xclang -fallow-half-arguments-and-returns -D__HIP_HCC_COMPAT_MODE__=1";
    }

    if (not $isWindows) {
        $HSA_PATH=$ENV{'HSA_PATH'} // "$ROCM_PATH/hsa";
        $HIPCXXFLAGS .= " -isystem $HSA_PATH/include";
        $HIPCFLAGS .= " -isystem $HSA_PATH/include";
    }

} elsif ($HIP_PLATFORM eq "nvidia") {
    $CUDA_PATH=$ENV{'CUDA_PATH'} // '/usr/local/cuda';
    $HIP_INCLUDE_PATH = "$HIP_PATH/include";
    if ($verbose & 0x2) {
        print ("CUDA_PATH=$CUDA_PATH\n");
    }

    $HIPCC="$CUDA_PATH/bin/nvcc";
    $HIPCXXFLAGS .= " -Wno-deprecated-gpu-targets ";
    $HIPCXXFLAGS .= " -isystem $CUDA_PATH/include";
    $HIPCFLAGS .= " -isystem $CUDA_PATH/include";

    $HIPLDFLAGS = " -Wno-deprecated-gpu-targets -lcuda -lcudart -L$CUDA_PATH/lib64";
} else {
    printf ("error: unknown HIP_PLATFORM = '$HIP_PLATFORM'");
    printf ("       or HIP_COMPILER = '$HIP_COMPILER'");
    exit (-1);
}

# Add paths to common HIP includes:
$HIPCXXFLAGS .= " -isystem \"$HIP_INCLUDE_PATH\"" ;
$HIPCFLAGS .= " -isystem \"$HIP_INCLUDE_PATH\"" ;

my $compileOnly = 0;
my $needCXXFLAGS = 0;  # need to add CXX flags to compile step
my $needCFLAGS = 0;    # need to add C flags to compile step
my $needLDFLAGS = 1;   # need to add LDFLAGS to compile step.
my $fileTypeFlag = 0;  # to see if -x flag is mentioned
my $hasOMPTargets = 0; # If OMP targets is mentioned
my $hasC = 0;          # options contain a c-style file
my $hasCXX = 0;        # options contain a cpp-style file (NVCC must force recognition as GPU file)
my $hasCU = 0;         # options contain a cu-style file (HCC must force recognition as GPU file)
my $hasHIP = 0;        # options contain a hip-style file (HIP-Clang must pass offloading options)
my $printHipVersion = 0;    # print HIP version
my $printCXXFlags = 0;      # print HIPCXXFLAGS
my $printLDFlags = 0;       # print HIPLDFLAGS
my $runCmd = 1;
my $buildDeps = 0;
my $linkType = 1;
my $setLinkType = 0;
my $hsacoVersion = 0;
my $funcSupp = 0;      # enable function support
my $rdc = 0;           # whether -fgpu-rdc is on

my @options = ();
my @inputs  = ();

if ($verbose & 0x4) {
    print "hipcc-args: ", join (" ", @ARGV), "\n";
}

# Handle code object generation
my $ISACMD="";
if($HIP_PLATFORM eq "nvidia"){
    $ISACMD .= "$HIP_PATH/bin/hipcc -ptx ";
    if($ARGV[0] eq "--genco"){
        foreach $isaarg (@ARGV[1..$#ARGV]){
            $ISACMD .= " ";
            $ISACMD .= $isaarg;
        }
        if ($verbose & 0x1) {
            print "hipcc-cmd: ", $ISACMD, "\n";
        }
        system($ISACMD) and die();
        exit(0);
    }
}

# TODO: convert toolArgs to an array rather than a string
my $toolArgs = "";  # arguments to pass to the clang or nvcc tool
my $optArg = ""; # -O args

# TODO: hipcc uses --amdgpu-target for historical reasons. It should be replaced
# by clang option --offload-arch.
my @targetOpts = ('--offload-arch=', '--amdgpu-target=');

my $targetsStr = "";
my $skipOutputFile = 0; # file followed by -o should not contibute in picking compiler flags
my $prevArg = ""; # previous argument

foreach $arg (@ARGV)
{
    # Save $arg, it can get changed in the loop.
    $trimarg = $arg;
    # TODO: figure out why this space removal is wanted.
    # TODO: If someone has gone to the effort of quoting the spaces to the shell
    # TODO: why are we removing it here?
    $trimarg =~ s/^\s+|\s+$//g;  # Remive whitespace
    my $swallowArg = 0;
    my $escapeArg = 1;
    if ($arg eq '-c' or $arg eq '--genco' or $arg eq '-E') {
        $compileOnly = 1;
        $needLDFLAGS  = 0;
    }

    if ($skipOutputFile) {
	# TODO: handle filename with shell metacharacters
        $toolArgs .= " \"$arg\"";
        $prevArg = $arg;
        $skipOutputFile = 0;
        next;
    }

    if ($arg eq '-o') {
        $needLDFLAGS = 1;
        $skipOutputFile = 1;
    }

    if(($trimarg eq '-stdlib=libc++') and ($setStdLib eq 0))
    {
        $HIPCXXFLAGS .= " -stdlib=libc++";
        $setStdLib = 1;
    }

    # Check target selection option: --offload-arch= and --amdgpu-target=...
    foreach my $targetOpt (@targetOpts) {
        if (substr($arg, 0, length($targetOpt)) eq $targetOpt) {
             if ($targetOpt eq '--amdgpu-target=') {
                print "Warning: The --amdgpu-target option has been deprecated and will be removed in the future.  Use --offload-arch instead.\n";
             }
             # If targets string is not empty, add a comma before adding new target option value.
             $targetsStr .= ($targetsStr ? ',' : '');
             $targetsStr .=  substr($arg, length($targetOpt));
             $default_amdgpu_target = 0;
             # Collect the GPU arch options and pass them to clang later.
             if ($HIP_PLATFORM eq "amd") {
                 $swallowArg = 1;
             }
        }
    }

    if (($arg =~ /--genco/) and  $HIP_PLATFORM eq 'amd' ) {
        $arg = "--cuda-device-only";
    }

    if($trimarg eq '--version') {
        $printHipVersion = 1;
    }
    if($trimarg eq '--short-version') {
        $printHipVersion = 1;
        $runCmd = 0;
    }
    if($trimarg eq '--cxxflags') {
        $printCXXFlags = 1;
        $runCmd = 0;
    }
    if($trimarg eq '--ldflags') {
        $printLDFlags = 1;
        $runCmd = 0;
    }
    if($trimarg eq '-M') {
        $compileOnly = 1;
        $buildDeps = 1;
    }
    if(($trimarg eq '-use-staticlib') and ($setLinkType eq 0))
    {
        $linkType = 0;
        $setLinkType = 1;
        $swallowArg = 1;
    }
    if(($trimarg eq '-use-sharedlib') and ($setLinkType eq 0))
    {
        $linkType = 1;
        $setLinkType = 1;
    }
    if($arg =~ m/^-O/)
    {
        $optArg = $arg;
    }
    if($arg =~ '--amdhsa-code-object-version=')
    {
        print "Warning: The --amdhsa-code-object-version option has been deprecated and will be removed in the future.  Use -mllvm -mcode-object-version instead.\n";
        $arg =~ s/--amdhsa-code-object-version=//;
        $hsacoVersion = $arg;
        $swallowArg = 1;
    }

    # nvcc does not handle standard compiler options properly
    # This can prevent hipcc being used as standard CXX/C Compiler
    # To fix this we need to pass -Xcompiler for options
    if (($arg eq '-fPIC' or $arg =~ '-Wl,') and $HIP_COMPILER eq 'nvcc')
    {
        $HIPCXXFLAGS .= " -Xcompiler ".$arg;
        $swallowArg = 1;
    }

    ## process linker response file for hip-clang
    ## extract object files from static library and pass them directly to
    ## hip-clang in command line.
    ## ToDo: Remove this after hip-clang switch to lto and lld is able to
    ## handle clang-offload-bundler bundles.
    if (($arg =~ m/^-Wl,@/ or $arg =~ m/^@/) and
        $HIP_PLATFORM eq 'amd') {
        my @split_arg = (split /\@/, $arg); # arg will have options type(-Wl,@ or @) and filename
        my $file = $split_arg[1];
        open my $in, "<:encoding(utf8)", $file or die "$file: $!";
        my $new_arg = "";
        my $tmpdir = get_temp_dir ();
        my $new_file = "$tmpdir/response_file";
        open my $out, ">", $new_file or die "$new_file: $!";
        while (my $line = <$in>) {
            chomp $line;
            if ($line =~ m/\.a$/ || $line =~ m/\.lo$/) {
                my $libFile = $line;
                my $path = abs_path($line);
                my @objs = split ('\n', `cd $tmpdir; ar xv $path`);
                ## Check if all files in .a are object files.
                my $allIsObj = 1;
                my $realObjs = "";
                foreach my $obj (@objs) {
                    chomp $obj;
                    $obj =~ s/^x - //;
                    $obj = "$tmpdir/$obj";
                    my $fileType = `file $obj`;
                    my $isObj = ($fileType =~ m/ELF/ or $fileType =~ m/COFF/);
                    $allIsObj = ($allIsObj and $isObj);
                    if ($isObj) {
                        $realObjs = ($realObjs . " " . $obj);
                    } else {
                        push (@inputs, $obj);
                        $new_arg = "$new_arg $obj";
                    }
                }
                chomp $realObjs;
                if ($allIsObj) {
                    print $out "$line\n";
                } elsif ($realObjs) {
                    my($libBaseName, $libDir, $libExt) = fileparse($libFile);
                    $libBaseName = mktemp($libBaseName . "XXXX") . $libExt;
                    system("cd $tmpdir; ar rc $libBaseName $realObjs");
                    print $out "$tmpdir/$libBaseName\n";
                }
            } elsif ($line =~ m/\.o$/) {
                my $fileType = `file $line`;
                my $isObj = ($fileType =~ m/ELF/ or $fileType =~ m/COFF/);
                if ($isObj) {
                    print $out "$line\n";
                } else {
                    push (@inputs, $line);
                    $new_arg = "$new_arg $line";
                }
            } else {
                print $out "$line\n";
            }
        }
        close $in;
        close $out;
        $arg = "$new_arg $split_arg[0]\@$new_file";
        $escapeArg = 0;
    } elsif (($arg =~ m/\.a$/ || $arg =~ m/\.lo$/) &&
              $HIP_PLATFORM eq 'amd') {
        ## process static library for hip-clang
        ## extract object files from static library and pass them directly to
        ## hip-clang.
        ## ToDo: Remove this after hip-clang switch to lto and lld is able to
        ## handle clang-offload-bundler bundles.
        my $new_arg = "";
        my $tmpdir = get_temp_dir ();
        my $libFile = $arg;
        my $path = abs_path($arg);
        my @objs = split ('\n', `cd $tmpdir; ar xv $path`);
        ## Check if all files in .a are object files.
        my $allIsObj = 1;
        my $realObjs = "";
        foreach my $obj (@objs) {
            chomp $obj;
            $obj =~ s/^x - //;
            $obj = "$tmpdir/$obj";
            my $fileType = `file $obj`;
            my $isObj = ($fileType =~ m/ELF/ or $fileType =~ m/COFF/);
            if ($fileType =~ m/ELF/) {
                my $sections = `readelf -e -W $obj`;
                $isObj = !($sections =~ m/__CLANG_OFFLOAD_BUNDLE__/);
            }
            $allIsObj = ($allIsObj and $isObj);
            if ($isObj) {
                $realObjs = ($realObjs . " " . $obj);
            } else {
                push (@inputs, $obj);
                if ($new_arg ne "") {
                    $new_arg .= " ";
                }
                $new_arg .= "$obj";
            }
        }
        chomp $realObjs;
        if ($allIsObj) {
            $new_arg = $arg;
        } elsif ($realObjs) {
            my($libBaseName, $libDir, $libExt) = fileparse($libFile);
            $libBaseName = mktemp($libBaseName . "XXXX") . $libExt;
            system("cd $tmpdir; ar rc $libBaseName $realObjs");
            $new_arg .= " $tmpdir/$libBaseName";
        }
        $arg = "$new_arg";
        $escapeArg = 0;
        if ($toolArgs =~ m/-Xlinker$/) {
            $toolArgs = substr $toolArgs, 0, -8;
            chomp $toolArgs;
        }
    } elsif ($arg eq '-x') {
        $fileTypeFlag = 1;
    } elsif (($arg eq 'c' and $prevArg eq '-x') or ($arg eq '-xc')) {
        $fileTypeFlag = 1;
        $hasC = 1;
        $hasCXX = 0;
        $hasHIP = 0;
    } elsif (($arg eq 'c++' and $prevArg eq '-x') or ($arg eq '-xc++')) {
        $fileTypeFlag = 1;
        $hasC = 0;
        $hasCXX = 1;
        $hasHIP = 0;
    } elsif (($arg eq 'hip' and $prevArg eq '-x') or ($arg eq '-xhip')) {
        $fileTypeFlag = 1;
        $hasC = 0;
        $hasCXX = 0;
        $hasHIP = 1;
    } elsif ($arg  =~ '-fopenmp-targets=') {
        $hasOMPTargets = 1;
    } elsif ($arg =~ m/^-/) {
        # options start with -
        if  ($arg eq '-fgpu-rdc') {
            $rdc = 1;
        } elsif ($arg eq '-fno-gpu-rdc') {
            $rdc = 0;
        }

        # Process HIPCC options here:
        if ($arg =~ m/^--hipcc/) {
            $swallowArg = 1;
            if ($arg eq "--hipcc-func-supp") {
              print "Warning: The --hipcc-func-supp option has been deprecated and will be removed in the future.\n";
              $funcSupp = 1;
            } elsif ($arg eq "--hipcc-no-func-supp") {
              print "Warning: The --hipcc-no-func-supp option has been deprecated and will be removed in the future.\n";
              $funcSupp = 0;
            }
        } else {
            push (@options, $arg);
        }
        #print "O: <$arg>\n";
    } elsif ($prevArg ne '-o') {
        # input files and libraries
        # Skip guessing if `-x {c|c++|hip}` is already specified.

        # Add proper file extension before each file type
        # File Extension                 -> Flag
        # .c                             -> -x c
        # .cpp/.cxx/.cc/.cu/.cuh/.hip    -> -x hip
        if ($fileTypeFlag eq 0) {
            if ($arg =~ /\.c$/) {
                $hasC = 1;
                $needCFLAGS = 1;
                $toolArgs .= " -x c";
            } elsif (($arg =~ /\.cpp$/) or ($arg =~ /\.cxx$/) or ($arg =~ /\.cc$/) or ($arg =~ /\.C$/)) {
                $needCXXFLAGS = 1;
                if ($HIP_COMPILE_CXX_AS_HIP eq '0' or $HIP_PLATFORM ne "amd" or $hasOMPTargets eq 1) {
                    $hasCXX = 1;
                } elsif ($HIP_PLATFORM eq "amd") {
                    $hasHIP = 1;
                    $toolArgs .= " -x hip";
                }
            } elsif ((($arg =~ /\.cu$/ or $arg =~ /\.cuh$/) and $HIP_COMPILE_CXX_AS_HIP ne '0') or ($arg =~ /\.hip$/)) {
                $needCXXFLAGS = 1;
                if ($HIP_PLATFORM eq "amd") {
                    $hasHIP = 1;
                    $toolArgs .= " -x hip";
                } else {
                    $hasCU = 1;
                }
            }
        }
        if ($hasC) {
            $needCFLAGS = 1;
        } elsif ($hasCXX or $hasHIP) {
            $needCXXFLAGS = 1;
        }
        push (@inputs, $arg);
        #print "I: <$arg>\n";
    }
    # Produce a version of $arg where characters significant to the shell are
    # quoted. One could quote everything of course but don't bother for
    # common characters such as alphanumerics.
    # Do the quoting here because sometimes the $arg is changed in the loop
    # Important to have all of '-Xlinker' in the set of unquoted characters.
    if (not $isWindows and $escapeArg) {
        $arg =~ s/[^-a-zA-Z0-9_=+,.\/]/\\$&/g;
    }
    if ($isWindows and $escapeArg) {
        $arg =~ s/[^-a-zA-Z0-9_=+,.:\/\\]/\\$&/g;
    }
    $toolArgs .= " $arg" unless $swallowArg;
    $prevArg = $arg;
}

if($HIP_PLATFORM eq "amd"){
    # No AMDGPU target specified at commandline. So look for HCC_AMDGPU_TARGET
    if($default_amdgpu_target eq 1) {
        if (defined $ENV{HCC_AMDGPU_TARGET}) {
            $targetsStr = $ENV{HCC_AMDGPU_TARGET};
        } elsif (not $isWindows) {
            # Else try using rocm_agent_enumerator
            $ROCM_AGENT_ENUM = "${ROCM_PATH}/bin/rocm_agent_enumerator";
            $targetsStr = `${ROCM_AGENT_ENUM} -t GPU`;
            $targetsStr =~ s/\n/,/g;
        }
        $default_amdgpu_target = 0;
    }

    # Parse the targets collected in targetStr and set corresponding compiler options.
    my @targets = split(',', $targetsStr);
    $GPU_ARCH_OPT = " --offload-arch=";

    foreach my $val (@targets) {
        # Ignore 'gfx000' target reported by rocm_agent_enumerator.
        if ($val ne 'gfx000') {
            my @procAndFeatures = split(':', $val);
            $len = scalar @procAndFeatures;
            my $procName;
            if($len ge 1 and $len le 3) { # proc and features
                $procName = $procAndFeatures[0];
                for my $i (1 .. $#procAndFeatures) {
                    if (grep($procAndFeatures[$i], @knownFeatures) eq 0) {
                        print "Warning: The Feature: $procAndFeatures[$i] is unknown. Correct compilation is not guaranteed.\n";
                    }
                }
            } else {
                $procName = $val;
            }
            $GPU_ARCH_ARG = $GPU_ARCH_OPT . $val;
            $HIPLDARCHFLAGS .= $GPU_ARCH_ARG;
            if ($HIP_PLATFORM eq 'amd' and $hasHIP) {
                $HIPCXXFLAGS .= $GPU_ARCH_ARG;
            }
        }
    }
    if ($hsacoVersion > 0) {
        if ($compileOnly eq 0) {
            $HIPLDFLAGS .= " -mcode-object-version=$hsacoVersion";
        } else {
            $HIPCXXFLAGS .= " -mcode-object-version=$hsacoVersion";
        }
    }

    # rocm_agent_enumerator failed! Throw an error and die if linking is required
    if ($default_amdgpu_target eq 1 and $compileOnly eq 0) {
        print "No valid AMD GPU target was either specified or found. Please specify a valid target using --offload-arch=<target>.\n" and die();
    }

    $ENV{HCC_EXTRA_LIBRARIES}="\n";
}

if ($hasCXX and $HIP_PLATFORM eq 'nvidia') {
    $HIPCXXFLAGS .= " -x cu";
}

if ($buildDeps and $HIP_PLATFORM eq 'nvidia') {
    $HIPCXXFLAGS .= " -M -D__CUDACC__";
    $HIPCFLAGS .= " -M -D__CUDACC__";
}

if ($buildDeps and $HIP_PLATFORM eq 'amd') {
    $HIPCXXFLAGS .= " --cuda-host-only";
}

# Add --hip-link only if it is compile only and -fgpu-rdc is on.
if ($rdc and !$compileOnly and $HIP_PLATFORM eq 'amd') {
    $HIPLDFLAGS .= " --hip-link";
    $HIPLDFLAGS .= $HIPLDARCHFLAGS;
}

# hipcc currrently requires separate compilation of source files, ie it is not possible to pass
# CPP files combined with .O files
# Reason is that NVCC uses the file extension to determine whether to compile in CUDA mode or
# pass-through CPP mode.

if ($HIP_PLATFORM eq "amd") {
    # Set default optimization level to -O3 for hip-clang.
    if ($optArg eq "") {
        $HIPCXXFLAGS .= " -O3";
        $HIPCFLAGS .= " -O3";
        $HIPLDFLAGS .= " -O3";
    }
    if (!$funcSupp and $optArg ne "-O0" and $hasHIP) {
        $HIPCXXFLAGS .= " -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false";
        if ($needLDFLAGS and not $needCXXFLAGS) {
            $HIPLDFLAGS .= " -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false";
        }
    }

    if ($hasHIP) {
        if ($DEVICE_LIB_PATH ne "$ROCM_PATH/amdgcn/bitcode") {
            $HIPCXXFLAGS .= " --hip-device-lib-path=\"$DEVICE_LIB_PATH\"";
        }
    }
    if (not $isWindows) {
        $HIPLDFLAGS .= " -lgcc_s -lgcc -lpthread -lm -lrt";
    }

    if (not $isWindows  and not $compileOnly) {
      if ($linkType eq 0) {
        $toolArgs = " -L$HIP_LIB_PATH -lamdhip64 -L$ROCM_PATH/lib -lhsa-runtime64 -ldl -lnuma " . ${toolArgs};
      } else {
        $toolArgs = ${toolArgs} . " -Wl,-rpath=$HIP_LIB_PATH:$ROCM_PATH/lib -lamdhip64 ";
      }
      # To support __fp16 and _Float16, explicitly link with compiler-rt
      $HIP_CLANG_BUILTIN_LIB="$HIP_CLANG_PATH/../lib/clang/$HIP_CLANG_VERSION/lib/$HIP_CLANG_TARGET/libclang_rt.builtins.a";
      if (-e $HIP_CLANG_BUILTIN_LIB) {
        $toolArgs .= " -L$HIP_CLANG_PATH/../lib/clang/$HIP_CLANG_VERSION/lib/$HIP_CLANG_TARGET -lclang_rt.builtins "
      } else {
        $toolArgs .= " -L$HIP_CLANG_PATH/../lib/clang/$HIP_CLANG_VERSION/lib/linux -lclang_rt.builtins-x86_64 "
      }
    }
}

if ($HIPCC_COMPILE_FLAGS_APPEND) {
    $HIPCXXFLAGS .= " $HIPCC_COMPILE_FLAGS_APPEND";
    $HIPCFLAGS .= " $HIPCC_COMPILE_FLAGS_APPEND";
}
if ($HIPCC_LINK_FLAGS_APPEND) {
    $HIPLDFLAGS .= " $HIPCC_LINK_FLAGS_APPEND";
}

# TODO: convert CMD to an array rather than a string
my $CMD="$HIPCC";

if ($needCFLAGS) {
    $CMD .= " $HIPCFLAGS";
}

if ($needCXXFLAGS) {
    $CMD .= " $HIPCXXFLAGS";
}

if ($needLDFLAGS and not $compileOnly) {
    $CMD .= " $HIPLDFLAGS";
}
$CMD .= " $toolArgs";

if ($verbose & 0x1) {
    print "hipcc-cmd: ", $CMD, "\n";
}

if ($printHipVersion) {
    if ($runCmd) {
        print "HIP version: "
    }
    print $HIP_VERSION, "\n";
}
if ($printCXXFlags) {
    print $HIPCXXFLAGS;
}
if ($printLDFlags) {
    print $HIPLDFLAGS;
}
if ($runCmd) {
    system ("$CMD");
    if ($? == -1) {
        print "failed to execute: $!\n";
        exit($?);
    }
    elsif ($? & 127) {
        printf "child died with signal %d, %s coredump\n",
        ($? & 127),  ($? & 128) ? 'with' : 'without';
        exit($?);
    }
    else {
         $CMD_EXIT_CODE = $? >> 8;
    }
    $? or delete_temp_dirs ();
    exit($CMD_EXIT_CODE);
}

# vim: ts=4:sw=4:expandtab:smartindent
