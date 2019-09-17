color_prompt=yes
force_color_prompt=yes

alias ls='ls --color=auto'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

if [ -f ~/.user.bashrc ]; then
  . ~/.user.bashrc
fi

export LANG=ja_JP.UTF-8
