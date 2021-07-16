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
badd +1 2layerKRR/rls2.py
badd +1 term://~/Documents/research/concatML//9402:/bin/bash
badd +13 ~/Documents/research/concatML/2layerKRR/compositeKRR.py
badd +4 .gitignore
badd +9 TODO
argglobal
%argdel
tabnew
tabrewind
edit 2layerKRR/rls2.py
set splitbelow splitright
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
exe '1resize ' . ((&lines * 30 + 25) / 50)
exe 'vert 1resize ' . ((&columns * 94 + 95) / 191)
exe '2resize ' . ((&lines * 30 + 25) / 50)
exe 'vert 2resize ' . ((&columns * 96 + 95) / 191)
exe '3resize ' . ((&lines * 15 + 25) / 50)
argglobal
balt .gitignore
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
let s:l = 269 - ((16 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 269
normal! 09|
wincmd w
argglobal
if bufexists("2layerKRR/rls2.py") | buffer 2layerKRR/rls2.py | else | edit 2layerKRR/rls2.py | endif
if &buftype ==# 'terminal'
  silent file 2layerKRR/rls2.py
endif
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
let s:l = 128 - ((21 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 128
let s:c = 80 - ((73 * winwidth(0) + 48) / 96)
if s:c > 0
  exe 'normal! ' . s:c . '|zs' . 80 . '|'
else
  normal! 080|
endif
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
let s:l = 15 - ((14 * winheight(0) + 7) / 15)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 15
normal! 0
wincmd w
exe '1resize ' . ((&lines * 30 + 25) / 50)
exe 'vert 1resize ' . ((&columns * 94 + 95) / 191)
exe '2resize ' . ((&lines * 30 + 25) / 50)
exe 'vert 2resize ' . ((&columns * 96 + 95) / 191)
exe '3resize ' . ((&lines * 15 + 25) / 50)
tabnext
edit ~/Documents/research/concatML/2layerKRR/compositeKRR.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 30 + 95) / 191)
exe '2resize ' . ((&lines * 22 + 25) / 50)
exe 'vert 2resize ' . ((&columns * 160 + 95) / 191)
exe '3resize ' . ((&lines * 23 + 25) / 50)
exe 'vert 3resize ' . ((&columns * 160 + 95) / 191)
argglobal
enew
file \[coc-explorer]-1
balt vimspector.Variables
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
balt vimspector.Variables
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
let s:l = 13 - ((5 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 13
let s:c = 40 - ((7 * winwidth(0) + 80) / 160)
if s:c > 0
  exe 'normal! ' . s:c . '|zs' . 40 . '|'
else
  normal! 040|
endif
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
let s:l = 1 - ((0 * winheight(0) + 11) / 23)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1
normal! 0
wincmd w
3wincmd w
exe 'vert 1resize ' . ((&columns * 30 + 95) / 191)
exe '2resize ' . ((&lines * 22 + 25) / 50)
exe 'vert 2resize ' . ((&columns * 160 + 95) / 191)
exe '3resize ' . ((&lines * 23 + 25) / 50)
exe 'vert 3resize ' . ((&columns * 160 + 95) / 191)
tabnext 2
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
nohlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
