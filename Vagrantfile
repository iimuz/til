# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/trusty64"

  config.vm.provider "virtualbox" do |vb|
    vb.gui = true
    vb.memory = "2048"
    vb.cpus = 2
    vb.customize [
      "modifyvm", :id,
      "--vram", "256",
      "--accelerate3d", "on",
      "--hwvirtex", "on",
      "--nestedpaging", "on",
      "--largepages", "on",
      "--ioapic", "on",
      "--pae", "on",
    ]
  end

  config.vm.provision "shell", privileged: false, inline: <<-SHELL
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends ubuntu-desktop
    sudo apt-get install -y --no-install-recommends ibus-mozc
    sudo apt-get install -y --no-install-recommends fonts-takao
    sudo reboot
  SHELL
end
