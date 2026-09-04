# frontend/Dockerfile
# Multi-stage production build for ReturnShield AI React frontend

# Stage 1: Build static assets using Node.js
FROM node:20-alpine AS build

WORKDIR /app

# Install npm dependencies
COPY package*.json ./
RUN npm ci

# Copy frontend source code
COPY . .

# Use Docker-specific env so VITE_API_URL="" — nginx proxy handles /health and /predict
RUN cp .env.docker .env && npm run build

# Stage 2: Serve static files with Nginx
FROM nginx:alpine

# Remove default Nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled static bundle from build stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
