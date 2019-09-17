color_prompt=yes
force_color_prompt=yes

alias grep='grep --color=auto'
alias egrep='egrep --color=auto'
alias fgrep='fgrep --color=auto'
alias ls='ls --color=auto'

if [ -f ~/.user.bashrc ]; then
  . ~/.user.bashrc
fi

export LESSCHARSET=utf-8

GPG_TTY=$(tty)
export GPG_TTY
alias gpg='gpg --pinentry-mode=loopback'
