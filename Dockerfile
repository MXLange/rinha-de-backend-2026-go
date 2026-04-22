FROM golang:1.26.2-alpine AS build

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN mkdir -p /app/resources-bin \
    && HNSW_M=16 HNSW_EF_SEARCH=2 go run ./cmd/preprocess

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /bin/rinha-api .

FROM alpine:3.21

WORKDIR /app

COPY --from=build /bin/rinha-api /app/rinha-api
COPY --from=build /app/resources-bin/references.bin /app/resources/references.bin
COPY --from=build /app/resources-bin/hnsw.bin /app/resources/hnsw.bin
COPY normalization.json /app/resources/normalization.json
COPY mcc_risk.json /app/resources/mcc_risk.json

EXPOSE 8080

ENTRYPOINT ["/app/rinha-api"]
