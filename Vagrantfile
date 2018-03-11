# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "debian/stretch64"
  config.vm.network :forwarded_port, guest: 22, host: 2222, id: "ssh", auto_correct: true
  config.vm.network :private_network, id: "private-network", type: "dhcp", ip: "192.168.34.0"
  config.vm.synced_folder "~/src", "/home/vagrant/src", owner: "vagrant", group: "vagrant"

  config.vm.provider "virtualbox" do |vb|
    vb.gui = false
    vb.memory = "1024"
    vb.cpus = 1
    vb.customize [
      "modifyvm", :id,
      "--vram", "8",
      "--accelerate3d", "off",
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

    # bash setting
    echo -e "\nif [ -f ~/.bashrc.local ]; then\n  . ~/.bashrc.local\nfi\n" >> ~/.bashrc
    wget https://raw.githubusercontent.com/iimuz/dotfiles/master/.bashrc -O ~/.bashrc.local

    # tools for development tools
    sudo apt-get install -y --no-install-recommends git && \
      wget https://raw.githubusercontent.com/iimuz/dotfiles/master/.gitconfig -O ~/.gitconfig
    sudo apt-get install -y --no-install-recommends tmux && \
      wget https://raw.githubusercontent.com/iimuz/dotfiles/master/.tmux.conf -O ~/.tmux.conf && \
      wget https://raw.githubusercontent.com/iimuz/dotfiles/master/.inputrc -O ~/.inputrc
    sudo apt-get install -y --no-install-recommends vim && \
      wget https://raw.githubusercontent.com/iimuz/dotfiles/master/.vimrc -O ~/.vimrc

    # tools for source code management
    sudo apt-get install -y --no-install-recommends peco
    sudo apt-get install -y --no-install-recommends unzip && \
      wget https://github.com/motemen/ghq/releases/download/v0.8.0/ghq_linux_amd64.zip && \
      unzip ghq_linux_amd64.zip -d ghq && \
      sudo mv ghq/ghq /usr/bin/ && \
      rm -rf ghq ghq_linux_amd64.zip .wget-hsts && \
      echo -e "\n[ghq]\n  root = ~/src\n" >> ~/.gitconfig.local

    # for gcp
    sudo apt-get install -y --no-install-recommends \
      curl \
      ssh

    # terraform
    curl https://releases.hashicorp.com/terraform/0.11.3/terraform_0.11.3_linux_amd64.zip -O && \
      unzip terraform_0.11.3_linux_amd64.zip && \
      sudo mv terraform /usr/local/bin/

    sudo reboot
  SHELL
end
