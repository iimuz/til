#!/bin/bash
# usage
function usage()
{
  local cmdname=`basename $0`
  echo "${cmdname} is a tool for ..."
  echo ""
  echo "Usage: ${cmdname} [command] [<options>]"
  echo ""
  echo "Command:"
  echo "  help, --help, -h: print ${cmdname} help."
  echo "  version, --version, -v: print ${cmdname} version."
  echo "  run: run unbinded script."
  echo ""
  echo "Options:"
  echo "  no option."
  echo ""
  echo "Version: `version`"
}

# version
function version()
{
  local cmdname=`basename $0`
  echo "${cmdname} version 0.1.0"
}

# 実行ファイル自身の場所を取得
basepath=`dirname "${0}"`
expr "${0}" : "/.*" > /dev/null || basepath=`(cd "${basepath}" && pwd)`

# subcommand
case ${1} in
  check)
    # IOMMU が enabled になっていれば OK
    dmesg | grep -E "DMAR|IOMMU"
    # vfio-pci が有効になっていれば OK
    dmesg | grep -i vfio
    # 該当する GPU の Kernel driver in use が vfio-pci になっていれば OK
    lspci -vs 0000
  ;;
  device_id)
    lspci -nn | grep -i nvidia
  ;;
  run)
    # update packages
    sudo -E apt update
    sudo -E apt upgrade -y
    sudo -E apt autoremove -y
    sudo -E apt clean

    # unbind gpu
    sudo sed -i -e 's/GRUB_CMDLINE_LINUX=""/GRUB_CMDLINE_LINUX="intel_iommu=on"/g' /etc/default/grub
    sudo grub-mkconfig -o /boot/grub/grub.cfg

    read -p "Input device ids: " device_ids
    sudo sh -c "echo \"options vfio-pci ids=${device_ids}\" >> /etc/modprobe.d/vfio.conf"

    sudo sh -c "echo 'vfio-pci' >> /etc/modules-load.d/vfio-pci.conf"
    sudo sh -c "echo 'vfio' >> /etc/initramfs-tools/modules"
    sudo sh -c "echo 'vfio_iommu_type1' >> /etc/initramfs-tools/modules"
    sudo sh -c "echo 'vfio_pci' >> /etc/initramfs-tools/modules"
    sudo sh -c "echo 'kvm' >> /etc/initramfs-tools/modules"
    sudo sh -c "echo 'kvm_intel' >> /etc/initramfs-tools/modules"
    sudo update-initramfs -u

    sudo sh -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
    sudo sh -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
    sudo update-initramfs -u
  ;;
  help|--help|-h)
    usage
  ;;
  version|--version|-v)
    version
  ;;
  *)
    echo "[ERROR] Invalid subcommand '${1}'"
    usage
    exit 1
  ;;
esac

exit 0
