FROM node:8-alpine

RUN set -x \
  && : "Install firebase cli tools" \
  && npm install -g firebase-tools

# settings for runtime emulator
ENV HOST 0.0.0.0
EXPOSE 5000

# settings for Firebase login
EXPOSE 9005

WORKDIR /workspace
CMD ["firebase"]

