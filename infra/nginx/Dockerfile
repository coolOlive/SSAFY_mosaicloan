FROM nginx:1.25

RUN rm -rf /usr/share/nginx/html/*
#
# ✅ 오직 빌드 결과만 복사
COPY --from=mosaic/frontend:deploy /dist /usr/share/nginx/html

COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
