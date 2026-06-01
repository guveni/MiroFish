# Build stage
FROM node:22-alpine AS build-stage
WORKDIR /app

# Copy package manifests from frontend first for caching
COPY frontend/package*.json ./frontend/

# Install dependencies in /app/frontend
WORKDIR /app/frontend
RUN npm ci

# Now move back to /app and copy the necessary code
WORKDIR /app
COPY locales ./locales
COPY frontend ./frontend

# Run build from /app/frontend
WORKDIR /app/frontend
ENV VITE_API_BASE_URL=/
RUN npm run build

# Production stage
FROM nginx:stable-alpine AS production-stage
COPY --from=build-stage /app/frontend/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
