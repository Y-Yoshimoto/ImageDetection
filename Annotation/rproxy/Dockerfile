# Nginx リバースプロキシ用コンテナ
FROM nginx:stable-bullseye

# リバースプロキシ設定ファイルをコンテナにコピー
COPY rproxy.conf /etc/nginx/conf.d/default.conf

# ポート番号を公開
EXPOSE 80

# コンテナ起動時に Nginx を起動
CMD ["nginx", "-g", "daemon off;"]
