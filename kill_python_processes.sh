#!/bin/bash
ps aux | grep python3.6 | awk '{print $2}' | xargs kill -9
