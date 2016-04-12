# .bash_profile

# Get the environment variables
if [ -f ~/.profile ]; then
	. ~/.profile
fi

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

export PATH=$PATH":$HOME/js247994/bin"
export PATH=$PATH:$HOME/bin
