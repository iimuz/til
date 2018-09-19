#!/bin/sh

# if root user
if  test "${USER_ID}:${GROUP_ID}" = "0:0"; then
  exec "$@"
fi

# change user id
if [ ${USER_ID} != $(id ${USER_NAME} -u) ]; then
  usermod -u ${USER_ID} ${USER_NAME}
fi

# change group id
if [ ${GROUP_ID} != $(id ${USER_NAME} -g) ]; then
  groupmod -g ${GROUP_ID} ${USER_NAME}
fi

chown ${USER_NAME}:${USER_NAME} ${SOURCE_DIR}

exec gosu ${USER_NAME} "$@"
