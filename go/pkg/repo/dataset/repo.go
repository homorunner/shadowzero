package dataset

import (
	"context"
	"ddt/pkg/repo"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/jackc/pgx/v5"
)

type Repo struct {
	db *pgx.Conn
}

func New(args repo.RepoArgs) *Repo {
	return &Repo{
		db: args.DB,
	}
}

func (r *Repo) Handle(ctx context.Context, action string, args []string) {
	switch action {
	case "push":
		r.push(ctx, args)
	case "pull":
		r.pull(ctx, args)
	}
}

func (r *Repo) push(ctx context.Context, args []string) {
	if len(args) < 1 {
		log.Fatal("not enough arguments for push")
	}

	datasetDir := args[0]
	files, err := os.ReadDir(datasetDir)
	if err != nil {
		log.Fatal(err)
	}

	for _, file := range files {
		if !file.Type().IsRegular() || file.Name()[0] != 'c' {
			continue
		}
		c_file, err := os.OpenFile(filepath.Join(datasetDir, file.Name()), os.O_RDONLY, 0644)
		if err != nil {
			log.Fatal(err)
		}
		c, _ := io.ReadAll(c_file)
		c_file.Close()
		v_file, err := os.OpenFile(filepath.Join(datasetDir, "v"+file.Name()[1:]), os.O_RDONLY, 0644)
		if err != nil {
			continue
		}
		v, _ := io.ReadAll(v_file)
		v_file.Close()
		p_file, err := os.OpenFile(filepath.Join(datasetDir, "p"+file.Name()[1:]), os.O_RDONLY, 0644)
		if err != nil {
			continue
		}
		p, _ := io.ReadAll(p_file)
		p_file.Close()
		var id int
		err = r.db.QueryRow(ctx, "INSERT INTO dataset (train_id, c, v, p) VALUES ('', $1, $2, $3) RETURNING id", c, v, p).Scan(&id)
		if err != nil {
			log.Fatal(err)
		}

		log.Print("push dataset to db ok. id = ", id)

		os.Remove(filepath.Join(datasetDir, file.Name()))
		os.Remove(filepath.Join(datasetDir, "v"+file.Name()[1:]))
		os.Remove(filepath.Join(datasetDir, "p"+file.Name()[1:]))
	}
}

func (r *Repo) pull(ctx context.Context, args []string) {
	if len(args) < 1 {
		log.Fatal("not enough arguments for pull")
	}

	path := args[0]
	err := os.MkdirAll(path, 0644)
	if err != nil {
		log.Fatal(err)
	}

	tx, err := r.db.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		log.Fatal(err)
	}

	for func() bool {
		rows, err := tx.Query(ctx, "SELECT c, v, p FROM dataset ORDER BY id DESC LIMIT 10")
		if err != nil {
			log.Fatal(err)
		}
		defer rows.Close()

		empty := true

		for rows.Next() {
			var c, v, p []byte
			err := rows.Scan(&c, &v, &p)
			if err != nil {
				log.Fatal(err)
			}

			empty = false

			suffix := fmt.Sprintf("_%d_%d.pt", rand.Intn(12000), rand.Intn(12000))

			cFile := filepath.Join(path, "c"+suffix)
			if err := os.WriteFile(cFile, c, 0644); err != nil {
				log.Fatal(err)
			}
			vFile := filepath.Join(path, "v"+suffix)
			if err := os.WriteFile(vFile, v, 0644); err != nil {
				log.Fatal(err)
			}
			pFile := filepath.Join(path, "p"+suffix)
			if err := os.WriteFile(pFile, p, 0644); err != nil {
				log.Fatal(err)
			}

			log.Print("pulled dataset *" + suffix)
		}

		rows.Close()

		_, err = tx.Exec(ctx, "DELETE FROM dataset WHERE id IN (SELECT id FROM dataset ORDER BY id DESC LIMIT 10)")
		if err != nil {
			log.Fatal(err)
		}

		return !empty
	}() {
	}

	tx.Commit(ctx)
}
