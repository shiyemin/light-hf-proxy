#!/bin/sh

python -m build && twine upload dist/*
