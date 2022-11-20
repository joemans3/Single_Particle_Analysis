#!/bin/bash
set -e
set -x

cd "$HOME"

# Use as: bash setup.sh SETUP_VAR1=y SETUP_VAR2=y
#
# Here are the available variables:
#
# SETUP_CYGWIN_INSTALL: install cygwin packages (Windows)
# SETUP_BREW_INSTALL: install brew packages (Mac OS X)
# SETUP_APT_MISC: install packages via apt (Debian GNU+Linux)
# SETUP_APT_FOSSIL_BUILDDEP: install fossil build dependency
#   packages via apt (Debian GNU+Linux)
# SETUP_BUILD_FOSSIL: build fossil locally and install it
# SETUP_CHSH: change shell to ZSH (all except Windows)
#
# Note that this script will not overwrite files by default.

for x in "$@"; do
    eval "$x"
done

uname_out="$(uname -s)"
case "${uname_out}" in
    Linux*)  OS=Linux;;
    Darwin*) OS=Mac;;
    CYGWIN*) OS=Cygwin;;
    MINGW*)  OS=MinGw;;
    *) echo "invalid OS $uname_out"; exit 1;;
esac

command_exists() { which "$1" 2>&1 >/dev/null; }

install_contents_if_not_exists() {
    if ! [ -e "$1" ]; then
        cat > "$1".new
        if [ -n "$2" ]; then
            chmod "$2" "$1.new"
        fi
        mv "$1".new "$1"
    else
        cat > /dev/null
    fi
}

# ensure ~/bin/ exists
mkdir -p bin/

if [ "$OS" = Cygwin ] && [ "x$SETUP_CYGWIN_INSTALL" = xy ]; then

    if ! id -G -n | egrep -q '\bAdministrators\b'; then
        echo "This script must be run as an administrator."
        exit 1
    fi

    # the cygwin installer actually contains a package manager
    SETUP=/usr/local/bin/cygwin-setup.exe
    if ! [ -e "$SETUP" ]; then
        f=`mktemp`
        wget -O "$f" "https://www.cygwin.com/setup-x86_64.exe"
        install -v "$f" "$SETUP"
        rm "$f"
    fi

    # install packages
    cygwin-setup \
        -q \
        -P chere,zsh,git,python3,python3-numpy,unzip,zip,tar,rsync,openssh,gnupg2,nano || true
fi

if [ "$OS" = Cygwin ]; then
    CYGWIN_BASH_WPATH="$(cygpath -w /bin/bash)"
fi

if [ "$OS" = Mac ] && [ "x$SETUP_BREW_INSTALL" = xy ]; then
    if ! command_exists brew; then
        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    fi
    brew install \
         fossil ssh-copy-id openssh wget \
         zsh zsh-completions
    brew cask install osxfuse
    brew install sshfs
fi

if [ "x$SETUP_APT_MISC" = xy ]; then
    sudo apt install \
         zsh git unzip kdiff3 kate \
         wget openssh-client rsync sshfs
fi

if [ "x$SETUP_APT_FOSSIL_BUILDDEP" = xy ]; then
    sudo apt install curl build-essential libz-dev libssl-dev tcl-dev
fi

if [ "x$SETUP_BUILD_FOSSIL" = xy ]; then
    fossil_build=.fossil-build-tmp
    rm -rf "$fossil_build"
    mkdir "$fossil_build"
    pushd "$fossil_build"
    curl -o fossil.tgz "https://fossil-scm.org/index.html/tarball/release/fossil.tar.gz"
    tar xzf fossil.tgz --strip-components 1
    ./configure \
        --disable-option-checking --json --with-th1-docs --with-th1-hooks \
        --with-tcl=1 --with-tcl-stubs --with-tcl-private-stubs \
        CPPFLAGS='-D_FORTIFY_SOURCE=2' \
        CFLAGS='-g -O2 -fPIE -fstack-protector-strong -Wformat -Werror=format-security' \
        LDFLAGS='-fPIE -pie -Wl,-z,relro -Wl,-z,now -Wl,--export-dynamic'
    make -j
    if ! sudo make install; then
        echo "WARNING: failed to install fossil systemwide, copying to ~/bin/fossil instead" >&2
        cp -av fossil "$HOME/bin/fossil"
    fi
    make clean
    popd
fi

omz=.oh-my-zsh
if ! [ -e "$omz" ]; then
    git clone https://github.com/robbyrussell/oh-my-zsh.git "$omz"
fi

# zsh config: zshrc
install_contents_if_not_exists .zshrc <<'EOF'
# If you come from bash you might have to change your $PATH.
export PATH=$HOME/bin:/usr/local/bin:$PATH

# Path to your oh-my-zsh installation.
export ZSH=$HOME/.oh-my-zsh

# Set name of the theme to load. Optionally, if you set this to "random"
# it'll load a random theme each time that oh-my-zsh is loaded.
# See https://github.com/robbyrussell/oh-my-zsh/wiki/Themes
ZSH_THEME="ecd1"

# Set list of themes to load
# Setting this variable when ZSH_THEME=random
# cause zsh load theme from this variable instead of
# looking in ~/.oh-my-zsh/themes/
# An empty array have no effect
# ZSH_THEME_RANDOM_CANDIDATES=( "robbyrussell" "agnoster" )

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion. Case
# sensitive completion must be off. _ and - will be interchangeable.
# HYPHEN_INSENSITIVE="true"

# Uncomment the following line to disable bi-weekly auto-update checks.
DISABLE_AUTO_UPDATE="true"

# Uncomment the following line to change how often to auto-update (in days).
# export UPDATE_ZSH_DAYS=13

# Uncomment the following line to disable colors in ls.
# DISABLE_LS_COLORS="true"

# Uncomment the following line to disable auto-setting terminal title.
# DISABLE_AUTO_TITLE="true"

# Uncomment the following line to enable command auto-correction.
# ENABLE_CORRECTION="true"

# Uncomment the following line to display red dots whilst waiting for completion.
COMPLETION_WAITING_DOTS="true"

# Uncomment the following line if you want to disable marking untracked files
# under VCS as dirty. This makes repository status check for large repositories
# much, much faster.
# DISABLE_UNTRACKED_FILES_DIRTY="true"

# Uncomment the following line if you want to change the command execution time
# stamp shown in the history command output.
# The optional three formats: "mm/dd/yyyy"|"dd.mm.yyyy"|"yyyy-mm-dd"
HIST_STAMPS="yyyy-mm-dd"

# Would you like to use another custom folder than $ZSH/custom?
ZSH_CUSTOM="$HOME/.oh-my-zsh-custom"

# Which plugins would you like to load? (plugins can be found in ~/.oh-my-zsh/plugins/*)
# Custom plugins may be added to ~/.oh-my-zsh/custom/plugins/
# Example format: plugins=(rails git textmate ruby lighthouse)
# Add wisely, as too many plugins slow down shell startup.
plugins=(
  fossil command-not-found history

  ## this is a set of more advanced aliases, uncomment the following
  ## line only after looking at
  ## `~/.oh-my-zsh-custom/plugins/ecd-global-aliases/ecd-global-aliases.plugin.zsh`
  # ecd-global-aliases
)

source $ZSH/oh-my-zsh.sh

# User configuration

# export MANPATH="/usr/local/man:$MANPATH"

# You may need to manually set your language environment
# export LANG=en_US.UTF-8

# history size
HISTSIZE=40000
SAVEHIST=40000

# Preferred editor for local and remote sessions
if [[ -n $SSH_CONNECTION ]]; then
  export EDITOR="$HOME/bin/EDITOR"
# else
#   export EDITOR='mvim'
fi

# Compilation flags
# export ARCHFLAGS="-arch x86_64"

# ssh
# export SSH_KEY_PATH="~/.ssh/rsa_id"

# Set personal aliases, overriding those provided by oh-my-zsh libs,
# plugins, and themes. Aliases can be placed here, though oh-my-zsh
# users are encouraged to define aliases within the ZSH_CUSTOM folder.
# For a full list of active aliases, run `alias`.
#
# Example aliases
# alias zshconfig="mate ~/.zshrc"
# alias ohmyzsh="mate ~/.oh-my-zsh"
EOF

# add custom themes and plugins
ozc=".oh-my-zsh-custom"
mkdir -p "$ozc"
pushd "$ozc"

install_contents_if_not_exists aliases.zsh <<'EOF'
# Define your own aliases here!

alias f=fossil

EOF

install_contents_if_not_exists ecd-aliases.zsh <<'EOF'
# These are the aliases that came with the setup.sh script. I
# recommend putting your own custom aliases in `aliases.zsh` instead
# to ease future upgrades

alias diffp="diff -Naur" # produce path diff between files or directories

alias psr="ps -ef --sort=start_time" # all processes
alias p='ps -f' # running processes in current shell

alias no_history="unset HISTFILE" # incognito mode for shell history

alias ddp="dd status=progress" # dd with speed and progress info

alias sortn='sort -n -r' # sort numerically
alias sortnr='sort -n -r' # sort numerically, descending order

alias dud='du -d 1 -h' # print size for each directory (on first level)
alias duf='du -sh *' # print size for each file or directory (on first level)

alias fd='find . -type d -name' # search directories by name
alias ff='find . -type f -name' # search files by name

# ls, the common ones I use a lot shortened for rapid fire usage
# -A : all except . and ..
# -F : show type
# -h : human-readable
# -r : reverse
# -t : sort by date
# -l : long list
alias l='ls -lFh'
alias la='ls  -lFhA'
alias lt='ls  -lFhtr'
alias lta='ls -lFhtrA'
alias ll='ls -l'
alias ldot='ls -ld .*'
alias lS='ls -1FSshr'

# grep for source code trees
alias sgrep='grep -R -n -H -C 5 --exclude-dir={.git,.hg,.svn,CVS} '

# create directory and move to it
function mkcd() {
    [ -n "$1" ] && mkdir -p "$@" && cd "$1"
}

# search in zsh manpage
zman() {
  PAGER="less -g -s '+/^       "$1"'" man zshall
}

EOF

mkdir -p themes
install_contents_if_not_exists themes/ecd1.zsh-theme <<'EOF'

if [ -z "$zsh_theme_path_length" ]; then
    zsh_theme_path_length=60
fi

if [ -n "$SSH_CLIENT" ]; then
    zsh_theme_host_color="${fg_bold[red]}"; else
    zsh_theme_host_color="${fg[magenta]}"; fi

if [ "$(id -u)" -eq 0 ]; then
    zsh_theme_sep_color="${fg[green]}";
    zsh_theme_user_color="${fg_bold[red]}"; else
    zsh_theme_sep_color="${fg[green]}";
    zsh_theme_user_color="${fg[magenta]}"; fi

zsh_theme_prompt_info_prefix="%{$reset_color%}%{$fg[yellow]%}"

zsh_theme_prompt_info_commands=(virtualenv_prompt_info git_prompt_info fossil_prompt_info)

function ecd1_theme_combined_prompt_info() {
  local acc single prompt
  for prompt_command in $zsh_theme_prompt_info_commands; do
    if ! whence -w $prompt_command >/dev/null; then
      continue
    fi
    single="$(${prompt_command} | tr '\n' ' ' | sed -e 's/ \+$//;s/^ \+//;')"
    if [ -n "$single" ]; then
      if [ -n "$acc" ]; then
        acc="${acc}${zsh_theme_prompt_info_prefix} ${single}"
      else
        acc="${zsh_theme_prompt_info_prefix}${single}"
      fi
    fi
  done
  if [ -n "$acc" ]; then
    printf '%s' "%{$zsh_theme_sep_color%}(%{$reset_color%}${acc}%{$zsh_theme_sep_color%})%{$reset_color%}"
  fi
}

PROMPT='%{$zsh_theme_user_color%}%n%{$reset_color%}%{$zsh_theme_sep_color%}@%{$reset_color%}%{$zsh_theme_host_color%}%m%{$reset_color%}%{$zsh_theme_sep_color%}:%{$reset_color%}%{$fg_bold[default]%}%'"$zsh_theme_path_length"'<⋯<%~%<<%{$reset_color%}%{$zsh_theme_sep_color%}%(!.#.$)%{$reset_color%} '
RPROMPT='%{$reset_color%}$(ecd1_theme_combined_prompt_info)'

ZSH_THEME_GIT_PROMPT_PREFIX="g·%16>⋯>"
ZSH_THEME_GIT_PROMPT_SUFFIX=""
ZSH_THEME_GIT_PROMPT_DIRTY="%>>*"
ZSH_THEME_GIT_PROMPT_CLEAN="%>>"

ZSH_THEME_FOSSIL_PROMPT_PREFIX="f·%16>⋯>"
ZSH_THEME_FOSSIL_PROMPT_SUFFIX=""
ZSH_THEME_FOSSIL_PROMPT_DIRTY="%>>*"
ZSH_THEME_FOSSIL_PROMPT_CLEAN="%>>"

ZSH_THEME_VIRTUALENV_PREFIX="v·"
ZSH_THEME_VIRTUALENV_SUFFIX=" "

EOF

mkdir -p plugins
pushd plugins
mkdir -p fossil
install_contents_if_not_exists fossil/fossil.plugin.zsh <<'EOF'
_FOSSIL_PROMPT=""

# Prefix at the very beginning of the prompt, before the branch name
ZSH_THEME_FOSSIL_PROMPT_PREFIX="%{$fg_bold[blue]%}fossil:(%{$fg_bold[red]%}"

# At the very end of the prompt
ZSH_THEME_FOSSIL_PROMPT_SUFFIX="%{$fg_bold[blue]%})"

# Text to display if the branch is dirty
ZSH_THEME_FOSSIL_PROMPT_DIRTY=" %{$fg_bold[red]%}✖"

# Text to display if the branch is clean
ZSH_THEME_FOSSIL_PROMPT_CLEAN=" %{$fg_bold[green]%}✔"

function fossil_prompt_info () {
  local _OUTPUT
  _OUTPUT="$(fossil branch 2>&1)"
  if [ $? -eq 0 ]; then
    local _EDITED="$(fossil changes)"
    local _EDITED_SYM="$ZSH_THEME_FOSSIL_PROMPT_CLEAN"
    local _BRANCH="$(echo $_OUTPUT | sed -n 's/^\* \(.*\)$/\1/p')"

    if [ "$_EDITED" != "" ]; then
      _EDITED_SYM="$ZSH_THEME_FOSSIL_PROMPT_DIRTY"
    fi

    printf "%s%s%s%s%s" \
      "$ZSH_THEME_FOSSIL_PROMPT_PREFIX" \
      "$_BRANCH" \
      "$_EDITED_SYM"\
      "$ZSH_THEME_FOSSIL_PROMPT_SUFFIX" \
      "%{$reset_color%}"
  fi
}

function _fossil_get_command_list () {
  fossil help -a | grep -v "Usage|Common|This is"
}

function _fossil () {
  local context state state_descr line
  typeset -A opt_args

  _arguments \
    '1: :->command'\
    '2: :->subcommand'

  case $state in
    command)
      local _OUTPUT=`fossil branch 2>&1 | grep "use --repo"`
      if [ "$_OUTPUT" = "" ]; then
        compadd `_fossil_get_command_list`
      else
        compadd clone init import help version
      fi
      ;;
    subcommand)
      if [ "$words[2]" = "help" ]; then
        compadd `_fossil_get_command_list`
      else
        compcall -D
      fi
    ;;
  esac
}

function _fossil_prompt () {
  local current=`echo $PROMPT $RPROMPT | grep fossil`

  if [ "$_FOSSIL_PROMPT" = "" -o "$current" = "" ]; then
    local _prompt=${PROMPT}
    local _rprompt=${RPROMPT}

    local is_prompt=`echo $PROMPT | grep git`

    if [ "$is_prompt" = "" ]; then
      export RPROMPT="$_rprompt"'$(fossil_prompt_info)'
    else
      export PROMPT="$_prompt"'$(fossil_prompt_info) '
    fi

    _FOSSIL_PROMPT="1"
  fi
}

# disable since it's hopelessly broken
#compdef _fossil fossil

autoload -U add-zsh-hook

#add-zsh-hook precmd _fossil_prompt
EOF

mkdir -p ecd-global-aliases
install_contents_if_not_exists ecd-global-aliases/ecd-global-aliases.plugin.zsh <<'EOF'
# Global aliases are expanded even if they do not occur in command
# position.
#
# For example, `dmesg L` will expand to `dmesg | less`.
#
alias -g H='| head'
alias -g T='| tail'
alias -g G='| grep'
alias -g L="| less"
alias -g LL="2>&1 | less"
alias -g CA="2>&1 | cat -A"
alias -g NE="2> /dev/null"
alias -g NUL="> /dev/null 2>&1"
alias -g P="| pygmentize"
alias -g PP="2>&1| pygmentize"
EOF

popd
popd

if [ "$OS" = Cygwin ]; then
    pushd bin

    # install 'kate_cygwrap' editor script
    install_contents_if_not_exists kate_cygwrap 0755 <<'EOF'
#!/bin/bash
EXE="C:/Program Files/Kate/bin/kate.exe"
c="$1"; shift

case "$c" in
    run)
        A=()
        for x in "$@"; do
            case "$x" in
                /*)
                    if [ -e "$(dirname "$x")" ]; then
                        x="$(cygpath -w "$x")"
                    fi ;;
            esac
            A+=("$x")
        done
        "$EXE" "${A[@]}"
        ;;
    check)
        [ -e "$(cygpath -u "$EXE")" ]
        exit $?
        ;;
esac
EOF

    # install 'npp_cygwrap' editor script
    install_contents_if_not_exists npp_cygwrap 0755 <<'EOF'
#!/bin/bash
EXE="C:/Program Files/Notepad++/notepad++.exe"
c="$1"; shift

case "$c" in
    run)
        A=()
        for x in "$@"; do
            case "$x" in
                /*)
                    if [ -e "$(dirname "$x")" ]; then
                        x="$(cygpath -w "$x")"
                    fi ;;
            esac
            A+=("$x")
        done
        "$EXE" "${A[@]}"
        ;;
    check)
        [ -e "$(cygpath -u "$EXE")" ]
        exit $?
        ;;
esac
EOF
    popd

    # set up "Open ZSH here" file explorer shortcut
    chere -i -s zsh -t mintty || true

    # install latest fossil
    FOSSILW=/usr/local/bin/fossilw.exe
    if ! [ -e "$FOSSILW" ]; then
        tmp="$(mktemp -d)"
        pushd "$tmp"
        wget -O fossil.zip \
             "https://www.fossil-scm.org/index.html/uv/fossil-w64-2.10.zip"
        unzip fossil.zip
        install -v fossil.exe "$FOSSILW"
        popd
        rm -rf "$tmp"
    fi

    # install latest version of wrapper script
    FOSSIL=/usr/local/bin/fossil
    rm -f "$FOSSIL"
    install_contents_if_not_exists "$FOSSIL" 0755 <<'EOF'
#!/bin/bash

# This wrapper converts absolute path arguments from cygwin to windows
# before passing them to fossil (which doesn't understand cygwin paths).

# If it screws up and converts an argument you didn't want to convert,
# use `fossilw` (the actual fossil binary) directly.

A=()
for x in "$@"; do
    case "$x" in
        /root/*|/home/*|/cygdrive/*)
            if [ -e "$(dirname "$x")" ]; then
                x="$(cygpath -w "$x")"
            fi ;;
    esac
    A+=("$x")
done
fossilw.exe "${A[@]}"
exit $?
EOF
fi

# editor script 'k'
install_contents_if_not_exists ~/bin/k 0755 <<'EOF'
#!/bin/bash

is_windows=
is_mac=
is_linux=
uname_s="$(uname -s)"
case "${uname_s}" in
    CYGWIN*|MINGW*) is_windows=y;;
    Darwin*) is_mac=y;;
    Linux*) is_linux=y;;
esac

detached() { nohup "$@" >/dev/null 2>/dev/null & }
command_exists() { which "$1" 2>&1 >/dev/null; return $?; }

guess_is_gui() {
    # we need to guess whether this is a graphical console or not

    if [ -n "$K_FORCE_CLI" ]; then
        return 1 # user forced CLI
    fi
    if [ -n "$K_FORCE_GUI" ]; then
        return 0 # user forced GUI
    fi
    if [ -n "$SSH_CLIENT" ]; then
        return 1 # remote connection, likely not graphical
    fi
    if [ -n "${DISPLAY}${WAYLAND_DISPLAY}" ]; then
        return 0 # found display server
    else
        if [ -n "$is_linux" ]; then
            return 1 # no display server
        fi
    fi
    if [ -n "$TERM_PROGRAM" ]; then
        return 0 # likely being run under a terminal app on Mac
    fi
    if [ -n "$is_windows" ]; then
        return 0 # likely graphical
    fi
    return 0 # default to graphical
}

if guess_is_gui; then
    if [ -n "$is_windows" ]; then
        for e in kate_cygwrap npp_cygwrap; do
            if command_exists "$e" && "$e" check; then
                detached "$e" run "$@"
                exit $?
            fi
        done
    else
        for e in kate atom gedit kwrite leafpad; do
            if command_exists "$e"; then
                detached "$e" "$@"
                exit $?
            fi
        done
    fi
fi

# fall back to nano
exec nano "$@"

EOF

install_contents_if_not_exists ~/bin/EDITOR 0755 <<'EOF'
#!/bin/bash

while true; do
    k "$@"
    read -n 1 -s -p "*** press 'y' when done, or any other key to bring back the editor *** " yn
    echo
    case "$yn" in
        y|Y) break ;;
    esac
done
EOF

if [ -n "$SETUP_FOSSIL_SETTINGS" ]; then
    # configure fossil
    if [ "$OS" = Cygwin ]; then
        KDIFF3="C:/Program Files/KDiff3/kdiff3.exe"
        fossil settings --global editor "\"$CYGWIN_BASH_WPATH\" EDITOR"
    else
        KDIFF3=kdiff3

        if command_exists leafpad; then # tiny text editor for GNU+Linux
            fossil settings --global editor "leafpad"
        else
            fossil settings --global editor "EDITOR"
        fi
    fi
    fossil settings --global gdiff-command "\"$KDIFF3\""
    fossil settings --global gmerge-command "\"$KDIFF3\" \"%baseline\" \"%original\" \"%merge\" -o \"%output\""
fi

# kate configuration
if [ "$OS" = Cygwin ]; then
    # TODO: don't hardcode this path
    kate_config_dir="$(cygpath -u "$USERPROFILE")/Local Settings/"
else
    kate_config_dir="$HOME/.config/"
fi
mkdir -p "$kate_config_dir"
pushd "$kate_config_dir"
install_contents_if_not_exists katepartrc <<'EOF'
[Document]
Allow End of Line Detection=true
BOM=false
Backup Flags=0
Backup Prefix=
Backup Suffix=~
Encoding=UTF-8
End of Line=0
Indent On Backspace=true
Indent On Tab=true
Indent On Text Paste=false
Indentation Mode=normal
Indentation Width=4
Keep Extra Spaces=false
Line Length Limit=4096
Newline at End of File=true
On-The-Fly Spellcheck=false
Overwrite Mode=false
PageUp/PageDown Moves Cursor=false
Remove Spaces=0
ReplaceTabsDyn=true
Show Spaces=false
Show Tabs=true
Smart Home=true
Swap Directory=
Swap File Mode=1
Swap Sync Interval=15
Tab Handling=2
Tab Width=4
Word Wrap=false
Word Wrap Column=80

[Editor]
Encoding Prober Type=1
Fallback Encoding=ISO-8859-15

[Renderer]
Animate Bracket Matching=false
Schema=Normal
Show Indentation Lines=true
Show Whole Bracket Expression=true
Word Wrap Marker=false

[View]
Allow Mark Menu=true
Auto Brackets=false
Auto Center Lines=0
Auto Completion=true
Bookmark Menu Sorting=0
Default Mark Type=1
Dynamic Word Wrap=true
Dynamic Word Wrap Align Indent=80
Dynamic Word Wrap Indicators=1
Fold First Line=false
Folding Bar=true
Folding Preview=true
Icon Bar=false
Input Mode=0
Keyword Completion=true
Line Modification=true
Line Numbers=true
Maximum Search History Size=100
Persistent Selection=false
Scroll Bar Marks=false
Scroll Bar Mini Map All=false
Scroll Bar Mini Map Width=60
Scroll Bar MiniMap=true
Scroll Bar Preview=true
Scroll Past End=false
Search/Replace Flags=140
Show Scrollbars=0
Show Word Count=false
Smart Copy Cut=false
Vi Input Mode Steal Keys=false
Vi Relative Line Numbers=false
Word Completion=true
Word Completion Minimal Word Length=3
Word Completion Remove Tail=true
EOF

if [ "$OS" = Linux ]; then
    # default font on debian isn't monospace, wtf
    install_contents_if_not_exists kateschemarc <<'EOF'
[Normal]
Font=DejaVu Sans Mono,14,-1,5,50,0,0,0,0,0
dummy=prevent-empty-group

[Breeze Dark]
Font=DejaVu Sans Mono,14,-1,5,50,0,0,0,0,0
dummy=prevent-empty-group
EOF
fi

popd

if [ "x$SETUP_CHSH" = xy ]; then
    ZSH="$(which zsh)"
    if [ "$OS" = Mac ]; then
        sudo dscl . -create /Users/"$USER" UserShell "$ZSH"
    else
        chsh -s "$ZSH" || echo "failed to chsh, skipping"
    fi
fi

printf "\n*** SUCCESS ***\n\n"
