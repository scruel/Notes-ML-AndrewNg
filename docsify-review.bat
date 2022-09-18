@echo off
REM npm i -g docsify-cli
REM npm i -g https://github.com/scruel/docsify-tools

REM docsify init ./docs

call docsify-auto-sidebar -d .
docsify serve .
