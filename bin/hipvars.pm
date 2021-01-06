#!/usr/bin/perl -w
package hipvars;
use Getopt::Long;
use Cwd;
use File::Basename;

$HIP_BASE_VERSION_MAJOR = "4";
$HIP_BASE_VERSION_MINOR = "0";

#---
# Function to parse config file
sub parse_config_file {
    my ($file, $config) = @_;
    if (open (CONFIG, "$file")) {
        while (<CONFIG>) {
            my $config_line=$_;
            chop ($config_line);
            $config_line =~ s/^\s*//;
            $config_line =~ s/\s*$//;
            if (($config_line !~ /^#/) && ($config_line ne "")) {
                my ($name, $value) = split (/=/, $config_line);
                $$config{$name} = $value;
            }
        }
        close(CONFIG);
    }
}

#---
# Function to check if executable can be run
sub can_run {
    my ($exe) = @_;
    `$exe --version 2>&1`;
    if ($? == 0) {
        return 1;
    } else {
        return 0;
    }
}

#
# TODO: Fix rpath LDFLAGS settings
#
# Since this hipcc script gets installed at two uneven hierarchical levels,
# linked by symlink, the absolute path of this script should be used to
# derive HIP_PATH, as dirname $0 could be /opt/rocm/bin or /opt/rocm/hip/bin
# depending on how it gets invoked.
# ROCM_PATH which points to <rocm_install_dir> is determined based on whether
# we find bin/rocm_agent_enumerator in the parent of HIP_PATH or not. If it is found,
# ROCM_PATH is defined relative to HIP_PATH else it is hardcoded to /opt/rocm.
#
$HIP_PATH=$ENV{'HIP_PATH'} // dirname(Cwd::abs_path("$0/../")); # use parent directory of hipcc
if (-e "$HIP_PATH/../bin/rocm_agent_enumerator") {
    $ROCM_PATH=$ENV{'ROCM_PATH'} // dirname("$HIP_PATH"); # use parent directory of HIP_PATH
} else {
    $ROCM_PATH=$ENV{'ROCM_PATH'} // "/opt/rocm";
}
$CUDA_PATH=$ENV{'CUDA_PATH'} // '/usr/local/cuda';
$HSA_PATH=$ENV{'HSA_PATH'} // "$ROCM_PATH/hsa";
$HIP_CLANG_PATH=$ENV{'HIP_CLANG_PATH'} // "$ROCM_PATH/llvm/bin";
# HIP_ROCCLR_HOME is used by Windows builds
$HIP_ROCCLR_HOME=$ENV{'HIP_ROCCLR_HOME'};

if (defined $HIP_ROCCLR_HOME) {
    $HIP_INFO_PATH= "$HIP_ROCCLR_HOME/lib/.hipInfo";
} else {
    $HIP_INFO_PATH= "$HIP_PATH/lib/.hipInfo"; # use actual file
}
#---
#HIP_PLATFORM controls whether to use nvidia or amd platform:
$HIP_PLATFORM=$ENV{'HIP_PLATFORM'};
# Read .hipInfo
my %hipInfo = ();
parse_config_file("$HIP_INFO_PATH", \%hipInfo);
# Prioritize Env first, otherwise use the hipInfo config file
$HIP_COMPILER = $ENV{'HIP_COMPILER'} // $hipInfo{'HIP_COMPILER'} // "clang";
$HIP_RUNTIME = $ENV{'HIP_RUNTIME'} // $hipInfo{'HIP_RUNTIME'} // "rocclr";

# If using ROCclr runtime, need to find HIP_ROCCLR_HOME
if (defined $HIP_RUNTIME and $HIP_RUNTIME eq "rocclr" and !defined $HIP_ROCCLR_HOME) {
    my $hipvars_dir = dirname($0);
    if (-e "$hipvars_dir/../lib/bitcode") {
        $HIP_ROCCLR_HOME = abs_path($hipvars_dir . "/..");
    } else {
        $HIP_ROCCLR_HOME = $HIP_PATH; # use HIP_PATH
    }
}

if (not defined $HIP_PLATFORM) {
    if (can_run("$HIP_CLANG_PATH/clang++") or can_run("clang++")) {
        $HIP_PLATFORM = "amd";
    } elsif (can_run("$CUDA_PATH/bin/nvcc") or can_run("nvcc")) {
        $HIP_PLATFORM = "nvidia";
        $HIP_COMPILER = "nvcc";
        $HIP_RUNTIME = "cuda";
    } else {
        # Default to amd for now
        $HIP_PLATFORM = "amd";
    }
}

if ($HIP_COMPILER eq "clang") {
    # Windows does not have clang at linux default path
    if (defined $HIP_ROCCLR_HOME and (-e "$HIP_ROCCLR_HOME/bin/clang" or -e "$HIP_ROCCLR_HOME/bin/clang.exe")) {
        $HIP_CLANG_PATH = "$HIP_ROCCLR_HOME/bin";
    }
}

#---
# Read .hipVersion
my %hipVersion = ();
parse_config_file("$hipvars::HIP_PATH/bin/.hipVersion", \%hipVersion);
$HIP_VERSION_MAJOR = $hipVersion{'HIP_VERSION_MAJOR'} // $HIP_BASE_VERSION_MAJOR;
$HIP_VERSION_MINOR = $hipVersion{'HIP_VERSION_MINOR'} // $HIP_BASE_VERSION_MINOR;
$HIP_VERSION_PATCH = $hipVersion{'HIP_VERSION_PATCH'} // "0";
$HIP_VERSION="$HIP_VERSION_MAJOR.$HIP_VERSION_MINOR.$HIP_VERSION_PATCH";