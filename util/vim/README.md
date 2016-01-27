### How to install? ###
1. Add the <code>hip.vim</code> to <code>~/.vim/syntax/</code> directory
2. Add the following text to the end of <code>~/.vimrc</code>


```
augroup filetypedetect
 au BufNewFile,BufRead *.cpp set filetype=cpp syntax=hip
augroup END
augroup filetypedetect
 au BufNewFile,BufRead *.c set filetype=c syntax=hip
augroup END
augroup filetypedetect
 au BufNewFile,BufRead *.cu set filetype=cu syntax=hip
augroup END
```
