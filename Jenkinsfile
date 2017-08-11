#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

////////////////////////////////////////////////////////////////////////
// -- AUXILLARY HELPER FUNCTIONS

////////////////////////////////////////////////////////////////////////
// Construct the relative path of the build directory
String build_directory_rel( String build_config )
{
  if( build_config.equalsIgnoreCase( 'release' ) )
  {
    return "build/release"
  }
  else
  {
    return "build/debug"
  }
}

////////////////////////////////////////////////////////////////////////
// -- BUILD RELATED FUNCTIONS

////////////////////////////////////////////////////////////////////////
// Checkout source code, source dependencies and update version number numbers
// Returns a relative path to the directory where the source exists in the workspace
String checkout_and_version( String platform )
{
  String source_dir_rel = "src"
  String source_hip_rel = "${source_dir_rel}/hip"

  stage("${platform} clone")
  {
    dir( "${source_hip_rel}" )
    {
      // checkout hip
      checkout([
        $class: 'GitSCM',
        branches: scm.branches,
        doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
        extensions: scm.extensions + [[$class: 'CleanCheckout']],
        userRemoteConfigs: scm.userRemoteConfigs
      ])
    }
  }

  return source_hip_rel
}


////////////////////////////////////////////////////////////////////////
// This creates the docker image that we use to build the project in
// The docker images contains all dependencies, including OS platform, to build
def docker_build_image( String platform, String source_hip_rel, String from_image )
{
  String project = "hip"
  String build_type_name = "build-ubuntu-16.04"
  String dockerfile_name = "dockerfile-${build_type_name}"
  String build_image_name = "${build_type_name}"
  def build_image = null

  stage("${platform} build image")
  {
    dir("${source_hip_rel}")
    {
      def user_uid = sh( script: 'id -u', returnStdout: true ).trim()

      // Docker 17.05 introduced the ability to use ARG values in FROM statements
      // Docker inspect failing on FROM statements with ARG https://issues.jenkins-ci.org/browse/JENKINS-44836
      //build_image = docker.build( "${project}/${build_image_name}:latest", "--pull -f docker/${dockerfile_name} --build-arg user_uid=${user_uid} --build-arg base_image=${from_image} ." )

      // JENKINS-44836 workaround by using a bash script instead of docker.build()
      sh "docker build -t ${project}/${build_image_name}:latest --pull -f docker/${dockerfile_name} --build-arg user_uid=${user_uid} --build-arg base_image=${from_image} ."
      build_image = docker.image( "${project}/${build_image_name}:latest" )
    }
  }

  return build_image
}

////////////////////////////////////////////////////////////////////////
// This encapsulates the cmake configure, build and package commands
// Leverages docker containers to encapsulate the build in a fixed environment
def docker_build_inside_image( def build_image, String inside_args, String platform, String optional_configure, String build_config, String source_hip_rel, String build_dir_rel )
{
  String source_hip_abs = pwd() + "/" + source_hip_rel

  build_image.inside( inside_args )
  {
    stage("${platform} make ${build_config}")
    {
      // The rm command needs to run as sudo because the test steps below create files owned by root
      sh  """#!/usr/bin/env bash
          set -x
          sudo rm -rf ${build_dir_rel}
          mkdir -p ${build_dir_rel}
          cd ${build_dir_rel}
          cmake -DCMAKE_BUILD_TYPE=${build_config} -DCMAKE_INSTALL_PREFIX=staging ${optional_configure} ${source_hip_abs}
          make -j\$(nproc)
        """
    }

    // Cap the maximum amount of testing, in case of hangs
    timeout(time: 1, unit: 'HOURS')
    {
      stage("${platform} unit testing")
      {
        sh  """#!/usr/bin/env bash
            set -x
            cd ${build_dir_rel}
            make install -j\$(nproc)
            make build_tests -i -j\$(nproc)
            make test 
          """
        // If unit tests output a junit or xunit file in the future, jenkins can parse that file
        // to display test results on the dashboard
        // junit "${build_dir_rel}/*.xml"
      }
    }

    // Only create packages from hcc based builds
    if( platform.toLowerCase( ).startsWith( 'hcc-' ) )
    {
      stage("${platform} packaging")
      {
        sh  """#!/usr/bin/env bash
            set -x
            cd ${build_dir_rel}
            make package
          """

        // No matter the base platform, all packages have the same name
        // Only upload 1 set of packages, so we don't have a race condition uploading packages
        // I arbitrarily pick hcc-1.6 as the most stable
        if( platform.toLowerCase( ).startsWith( 'hcc-1.6' ) )
        {
          archiveArtifacts artifacts: "${build_dir_rel}/*.deb", fingerprint: true
          archiveArtifacts artifacts: "${build_dir_rel}/*.rpm", fingerprint: true
        }
      }
    }
  }

  return void
}

////////////////////////////////////////////////////////////////////////
// This builds a fresh docker image FROM a clean base image, with no build dependencies included
// Uploads the new docker image to internal artifactory
def docker_upload_artifactory( String hcc_ver, String from_image, String source_hip_rel, String build_dir_rel )
{
  def hip_install_image = null
  String image_name = "hip-${hcc_ver}-ubuntu-16.04"
  String artifactory_org = env.JOB_NAME.toLowerCase( )

  stage( 'artifactory' )
  {
    println "artifactory_org: ${artifactory_org}"

    //  We copy the docker files into the bin directory where the .deb lives so that it's a clean build everytime
    sh "cp -r ${source_hip_rel}/docker/* ${build_dir_rel}"

    // Docker 17.05 introduced the ability to use ARG values in FROM statements
    // Docker inspect failing on FROM statements with ARG https://issues.jenkins-ci.org/browse/JENKINS-44836
    // hip_install_image = docker.build( "${artifactory_org}/${image_name}:${env.BUILD_NUMBER}", "--pull -f ${build_dir_rel}/dockerfile-hip-ubuntu-16.04 --build-arg base_image=${from_image} ${build_dir_rel}" )

    // The --build-arg REPO_RADEON= is a temporary fix to get around a DNS issue with our build machines
    // JENKINS-44836 workaround by using a bash script instead of docker.build()
    sh "docker build -t ${artifactory_org}/${image_name}:${env.BUILD_NUMBER} --pull -f ${build_dir_rel}/dockerfile-hip-ubuntu-16.04 --build-arg base_image=${from_image} ${build_dir_rel}"
    hip_install_image = docker.image( "${artifactory_org}/${image_name}:${env.BUILD_NUMBER}" )

    // The connection to artifactory can fail sometimes, but this should not be treated as a build fail
    try
    {
      // Don't push pull requests to artifactory, these tend to accumulate over time
      if( env.BRANCH_NAME.toLowerCase( ).startsWith( 'pr-' ) )
      {
        println 'Pull Request (PR-xxx) detected; NOT pushing to artifactory'
      }
      else
      {
        docker.withRegistry('http://compute-artifactory:5001', 'artifactory-cred' )
        {
          hip_install_image.push( "${env.BUILD_NUMBER}" )
          hip_install_image.push( 'latest' )
        }
      }
    }
    catch( err )
    {
      currentBuild.result = 'SUCCESS'
    }

    // Lots of images with tags are created above; no apparent way to delete images:tags with docker global variable
    // run bash script to clean images:tags after successful pushing
    sh "docker images | grep \"${artifactory_org}/${image_name}\" | awk '{print \$1 \":\" \$2}' | xargs docker rmi"
  }
}

////////////////////////////////////////////////////////////////////////
// -- MAIN
// Following this line is the start of MAIN of this Jenkinsfile
String build_config = 'Release'

parallel hcc_ctu:
{
  node('docker && rocm && gfx803')
  {
    String hcc_ver = 'hcc-ctu'
    String from_image = 'compute-artifactory:5001/radeonopencompute/hcc/clang_tot_upgrade/hcc-lc-ubuntu-16.04:latest'
    String inside_args = '--device=/dev/kfd'

    // Checkout source code, dependencies and version files
    String source_hip_rel = checkout_and_version( hcc_ver )

    // Create/reuse a docker image that represents the hip build environment
    def hip_build_image = docker_build_image( hcc_ver, source_hip_rel, from_image )

    // Print system information for the log
    hip_build_image.inside( inside_args )
    {
      sh  """#!/usr/bin/env bash
          set -x
          /opt/rocm/bin/rocm_agent_enumerator -t ALL
          /opt/rocm/bin/hcc --version
        """
    }

    // Conctruct a binary directory path based on build config
    String build_hip_rel = build_directory_rel( build_config );

    // Build hip inside of the build environment
    docker_build_inside_image( hip_build_image, inside_args, hcc_ver, '', build_config, source_hip_rel, build_hip_rel )

    // After a successful build, upload a docker image of the results
    docker_upload_artifactory( hcc_ver, from_image, source_hip_rel, build_hip_rel )
  }
},
hcc_1_6:
{
  node('docker && rocm && gfx803')
  {
    String hcc_ver = 'hcc-1.6'
    String from_image = 'compute-artifactory:5001/radeonopencompute/hcc/roc-1.6.x/hcc-lc-ubuntu-16.04:latest'
    String inside_args = '--device=/dev/kfd'

    // Checkout source code, dependencies and version files
    String source_hip_rel = checkout_and_version( hcc_ver )

    // Create/reuse a docker image that represents the hip build environment
    def hip_build_image = docker_build_image( hcc_ver, source_hip_rel, from_image )

    // Print system information for the log
    hip_build_image.inside( inside_args )
    {
      sh  """#!/usr/bin/env bash
          set -x
          /opt/rocm/bin/rocm_agent_enumerator -t ALL
          /opt/rocm/bin/hcc --version
        """
    }

    // Conctruct a binary directory path based on build config
    String build_hip_rel = build_directory_rel( build_config );

    // Build hip inside of the build environment
    docker_build_inside_image( hip_build_image, inside_args, hcc_ver, '', build_config, source_hip_rel, build_hip_rel )

    // After a successful build, upload a docker image of the results
    docker_upload_artifactory( hcc_ver, from_image, source_hip_rel, build_hip_rel )
  }
},
nvcc:
{
  node('docker && cuda')
  {
    ////////////////////////////////////////////////////////////////////////
    // Block of string constants customizing behavior for cuda
    String nvcc_ver = 'nvcc-8.0'
    String from_image = 'nvidia/cuda:8.0-devel'

    // This unfortunately hardcodes the driver version nvidia_driver_375.66 in the volume mount.  Research if a way
    // exists to get volume driver to customize the volume names to leave out driver version
    String inside_args = '''--device=/dev/nvidiactl --device=/dev/nvidia0 --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools
        --volume-driver=nvidia-docker --volume=nvidia_driver_375.66:/usr/local/nvidia:ro''';

    // Checkout source code, dependencies and version files
    String source_hip_rel = checkout_and_version( nvcc_ver )

    // We pull public nvidia images
    def hip_build_image = docker_build_image( nvcc_ver, source_hip_rel, from_image )

    // Print system information for the log
    hip_build_image.inside( inside_args )
    {
      sh  """#!/usr/bin/env bash
          set -x
          nvidia-smi
          nvcc --version
        """
    }

    // Conctruct a binary directory path based on build config
    String build_hip_rel = build_directory_rel( build_config );

    // Build hip inside of the build environment
    docker_build_inside_image( hip_build_image, inside_args, nvcc_ver, "-DHIP_NVCC_FLAGS=--Wno-deprecated-gpu-targets", build_config, source_hip_rel, build_hip_rel )

    // Not pushing an Nvidia based HiP to artifactory at this time
  }
}
