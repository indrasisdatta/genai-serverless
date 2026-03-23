Use Docker to pull a ready-made image with PostgreSQL + pgvector:

docker run -d \
  --name pgvector-db \
  -p 5433:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -v pgdata:/var/lib/postgresql/data \
  ankane/pgvector

Connect from terminal:
docker exec -it pgvector-db psql -U postgres

Connect from pgAdmin:
Host: localhost
Port: 5433
User: postgres
Password: postgres

Enable extension: 
CREATE EXTENSION vector;

Verify:
SELECT * FROM pg_extension;


