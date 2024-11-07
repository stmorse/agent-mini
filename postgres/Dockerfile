# Start from the official Postgres 14 image
FROM postgres:14

# Install testing tools: 'iproute2' for 'ss', 'procps' for 'ps'
RUN apt-get update && apt-get install -y \
    iproute2 \
    procps \
 && rm -rf /var/lib/apt/lists/*

# RUN chown -R 27099:1132 /var/lib/postgresql/data && \
#     chmod -R 700 /var/lib/postgresql/data

# Set the same entrypoint and command as the original Postgres image
CMD ["postgres"]