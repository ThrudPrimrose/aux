#!/usr/bin/expect
spawn  "ssh -o ProxyCommand='ssh -W %h:%p ge69xij@lxhalle.in.tum.de' ge69xij@sccs-gpu-login.sccs.in.tum.de"
expect "ge69xij@lxhalle.in.tum.de's password:"
send "pureassnow_8aA;"
expect "ge69xij@sccs-gpu-login.sccs.in.tum.de's password:"
send "lolisareonlyforheadpatting_8aA;"

