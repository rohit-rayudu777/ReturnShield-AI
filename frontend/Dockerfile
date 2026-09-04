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

# Accept optional build argument VITE_API_URL for Render / production deployments
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# If VITE_API_URL build arg is provided, write it to .env; otherwise copy .env.docker (relative /health and /predict paths for local Nginx proxy)
RUN if [ -n "$VITE_API_URL" ]; then echo "VITE_API_URL=$VITE_API_URL" > .env; else cp .env.docker .env; fi && npm run build

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
