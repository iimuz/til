VM_NAME='WinDev1712Eval'
OVA_FILE="${VM_NAME}.ova"
BOX_FILE="${VM_NAME}.box"

VBoxManage import "${OVA_FILE}"
VBoxManage list vms
vagrant package --base "${VM_NAME}" --output $BOX_FILE
VBoxManage unregistervm "${VM_NAME}" --delete
