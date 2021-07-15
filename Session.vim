let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Documents/research/concatML
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +89 2layerKRR/rls2.py
badd +1 term://~/Documents/research/concatML//9402:/bin/bash
badd +16 ~/Documents/research/concatML/2layerKRR/compositeKRR.py
argglobal
%argdel
edit 2layerKRR/rls2.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 30 + 95) / 191)
exe '2resize ' . ((&lines * 30 + 25) / 50)
exe 'vert 2resize ' . ((&columns * 79 + 95) / 191)
exe '3resize ' . ((&lines * 30 + 25) / 50)
exe 'vert 3resize ' . ((&columns * 80 + 95) / 191)
exe '4resize ' . ((&lines * 15 + 25) / 50)
exe 'vert 4resize ' . ((&columns * 160 + 95) / 191)
argglobal
enew
file \[coc-explorer]-1
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
wincmd w
argglobal
balt ~/Documents/research/concatML/2layerKRR/compositeKRR.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 194 - ((18 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 194
normal! 03|
wincmd w
argglobal
if bufexists("~/Documents/research/concatML/2layerKRR/compositeKRR.py") | buffer ~/Documents/research/concatML/2layerKRR/compositeKRR.py | else | edit ~/Documents/research/concatML/2layerKRR/compositeKRR.py | endif
if &buftype ==# 'terminal'
  silent file ~/Documents/research/concatML/2layerKRR/compositeKRR.py
endif
balt 2layerKRR/rls2.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 16 - ((15 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 16
normal! 034|
wincmd w
argglobal
if bufexists("term://~/Documents/research/concatML//9402:/bin/bash") | buffer term://~/Documents/research/concatML//9402:/bin/bash | else | edit term://~/Documents/research/concatML//9402:/bin/bash | endif
if &buftype ==# 'terminal'
  silent file term://~/Documents/research/concatML//9402:/bin/bash
endif
balt 2layerKRR/rls2.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 249 - ((14 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 249
normal! 060|
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 30 + 95) / 191)
exe '2resize ' . ((&lines * 30 + 25) / 50)
exe 'vert 2resize ' . ((&columns * 79 + 95) / 191)
exe '3resize ' . ((&lines * 30 + 25) / 50)
exe 'vert 3resize ' . ((&columns * 80 + 95) / 191)
exe '4resize ' . ((&lines * 15 + 25) / 50)
exe 'vert 4resize ' . ((&columns * 160 + 95) / 191)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0&& getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOFAcI
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
