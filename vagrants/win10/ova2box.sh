OVA_FILE='MSEdge - Win10.ova'
VM_NAME='MSEdge - Win10'
BOX_FILE='MSEdge-Win10.box'

VBoxManage import "${OVA_FILE}"
VBoxManage list vms
vagrant package --base "${VM_NAME}" --output $BOX_FILE
VBoxManage unregistervm "${VM_NAME}" --delete
