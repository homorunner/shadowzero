package db

import (
	"context"
	"ddt/pkg/config"

	"github.com/jackc/pgx/v5"
)

func Connect(ctx context.Context, cfg *config.DBConfig) (*pgx.Conn, error) {
	conn, err := pgx.Connect(ctx, cfg.URL)
	if err != nil {
		return nil, err
	}
	err = conn.Ping(ctx)
	if err != nil {
		return nil, err
	}
	return conn, nil
}
