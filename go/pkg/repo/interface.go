package repo

import "github.com/jackc/pgx/v5"

type RepoArgs struct {
	DB *pgx.Conn
}
