kill -9 $(ps -ef | grep python | grep gaojh | grep -v grep | awk '{print $2}')
