# test-go

Implementacao em Go para a Rinha de Backend 2026.

## O que tem aqui

- API com `GET /ready` e `POST /fraud-score`
- Vetorizacao 14D conforme a especificacao
- Busca exata top-5 com distancia euclidiana
- `nginx` como load balancer
- `docker-compose.yml` pronto para subir duas instancias da API

## Arquivos esperados

O projeto espera estes arquivos no diretorio raiz:

- `references.json.gz`
- `normalization.json`
- `mcc_risk.json`

Esses arquivos sao copiados para a imagem em `/app/resources`.

## Rodando localmente

```bash
go run .
```

Por padrao a API sobe em `:8080`.

## Rodando com Docker Compose

```bash
docker compose pull
docker compose up
```

O load balancer responde em:

```text
http://localhost:9999
```

## Testando com REST Client

O arquivo [`requests.http`](./requests.http) contem:

- `GET /ready`
- payloads com scores esperados `0.0`, `0.2`, `0.4`, `0.6`, `0.8` e `1.0`
