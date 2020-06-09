VALIDATE=0
HELP=N
DEBUG=0
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -v|--validate)
    if [ ! -z "$2" ]; then
      VALIDATE="$2"
    fi
    shift # past argument
    shift # past value
    ;;
    -r|--rocblas)
    ROCBLAS="$2"
    shift # past argument
    shift # past value
    ;;
    -g|--debug)
    DEBUG="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    HELP="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ "$HELP" != "N" ] ; 
then
   printf "$(basename "$0") [-h] [-v 1] -- program to build GemmDriver\n\n

where:\n
    -h (--help)       show this help text\n
    -r (--rocblas)    flag to use local copy of rocblas by specifying the base directory\n
    -v (--validate)   flag to enable blis validation option (default to 0)\n
    -g (--debug)      flag to build with debug symbols (default to 0)\n"
   exit 1
fi

# script begins here

install_blis()
{
      # /etc/*-release files describe the system
    if [[ -e "/etc/os-release" ]]; then
      source /etc/os-release
    elif [[ -e "/etc/centos-release" ]]; then
      ID=$(cat /etc/centos-release | awk '{print tolower($1)}')
      VERSION_ID=$(cat /etc/centos-release | grep -oP '(?<=release )[^ ]*' | cut -d "." -f1)
    else
      echo "This script depends on the /etc/*-release files"
      exit 2
    fi

    #Download prebuilt AMD multithreaded blis
    if [[ ! -f "extern/blis/lib/libblis.so.2" && $VALIDATE != 0 ]]; then
      mkdir -p extern
      case "${ID}" in
          centos|rhel|sles|opensuse-leap)
              wget -nv -O extern/blis.tar.gz https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-centos-2.0.tar.gz
              ;;
          ubuntu)
              wget -nv -O extern/blis.tar.gz https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-ubuntu-2.0.tar.gz
              ;;
          *)
              echo "Unsupported OS for this script"
              wget -nv -O extern/blis.tar.gz https://github.com/amd/blis/releases/download/2.0/aocl-blis-mt-ubuntu-2.0.tar.gz
              ;;
      esac

      cd extern
      tar -xvf blis.tar.gz
      rm -rf blis
      mv amd-blis-mt blis
      rm blis.tar.gz
      cd blis/lib
      ln -sf libblis-mt.so libblis.so
      cd ../../..
    fi

    #Download prebuilt AMD flame
    if [[ ! -f "extern/flame/lib/libflame.so.2" && $VALIDATE != 0 ]]; then
      case "${ID}" in
          centos|rhel|sles|opensuse-leap)
              wget -nv -O extern/flame.tar.gz https://github.com/amd/libflame/releases/download/2.0/aocl-libflame-centos-2.0.tar.gz
              ;;
          ubuntu)
              wget -nv -O extern/flame.tar.gz https://github.com/amd/libflame/releases/download/2.0/aocl-libflame-ubuntu-2.0.tar.gz
              ;;
          *)
              echo "Unsupported OS for this script"
              wget -nv -O extern/flame.tar.gz https://github.com/amd/libflame/releases/download/2.0/aocl-libflame-ubuntu-2.0.tar.gz
              ;;
      esac

      cd extern
      tar -xvf flame.tar.gz
      rm -rf flame
      mv amd-libflame flame
      rm flame.tar.gz
      cd ../
    fi
}

install_blis

make VALIDATE=$VALIDATE ROCBLASPATH=$ROCBLAS DEBUG=$DEBUG
