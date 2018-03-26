# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/trusty64"
  config.vm.network :forwarded_port, guest: 22, host: 2222, id: "ssh", auto_correct: true
  config.vm.network :private_network, id: "private-network", type: "dhcp", ip: "192.168.34.0"

  config.vm.provider "virtualbox" do |vb|
    vb.gui = false
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

  if Vagrant.has_plugin?("vagrant-vbguest") then
    config.vbguest.auto_update = true
  end

  # プロキシ設定
  if Vagrant.has_plugin?("vagrant-proxyconf")
    config.proxy.http     = "#{ENV['HTTP_PROXY']}"
    config.proxy.https     = "#{ENV['HTTP_PROXY']}"
    config.proxy.no_proxy = "localhost,127.0.0.1"
  end

  config.vm.provision "shell", privileged: false, inline: <<-SHELL
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends ubuntu-desktop
    sudo apt-get install -y --no-install-recommends ibus-mozc
    sudo apt-get install -y --no-install-recommends fonts-takao
    sudo reboot
  SHELL
end
