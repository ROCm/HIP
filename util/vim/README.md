<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [How to install?](#how-to-install)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

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
