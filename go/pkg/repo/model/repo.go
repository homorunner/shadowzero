package model

import (
	"context"
	"ddt/pkg/repo"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/samber/lo"
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
	case "newbest":
		r.newbest(ctx, args)
	case "getbest":
		r.getbest(ctx, args)
	}
}

func (r *Repo) push(ctx context.Context, args []string) {
	if len(args) < 1 {
		log.Fatal("not enough arguments for push")
	}

	modelFile := args[0]
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		log.Fatal(err)
	}

	content, err := os.ReadFile(modelFile)
	if err != nil {
		log.Fatal(err)
	}

	var id int
	err = r.db.QueryRow(ctx, "INSERT INTO models (name, param, content) VALUES ($1, $2, $3) RETURNING id", filepath.Base(modelFile), "{}", content).Scan(&id)
	if err != nil {
		log.Fatal(err)
	}

	log.Print("push model to db ok. id = ", id)
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

	currentModels := []string{""}
	for _, file := range lo.Must(os.ReadDir(path)) {
		if file.Type().IsRegular() && strings.HasSuffix(file.Name(), "_traced.pt") {
			currentModels = append(currentModels, file.Name())
		}
	}

	rows, err := r.db.Query(ctx, "SELECT name, content FROM models WHERE name != ALL($1)", currentModels)
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()

	for rows.Next() {
		var name string
		var content []byte
		err := rows.Scan(&name, &content)
		if err != nil {
			log.Fatal(err)
		}

		modelFile := filepath.Join(path, name)
		if err := os.WriteFile(modelFile, content, 0644); err != nil {
			log.Fatal(err)
		}

		log.Print("pulled model ", modelFile)
	}
}

func (r *Repo) newbest(ctx context.Context, args []string) {
	if len(args) < 1 {
		log.Fatal("not enough arguments for newbest")
	}

	modelFile := args[0]
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		log.Fatal(err)
	}

	var id int
	err := r.db.QueryRow(ctx, "INSERT INTO bestmodel (train_id, name) VALUES ($1, $2) ON CONFLICT (train_id) DO UPDATE SET name=$2 RETURNING id", "", filepath.Base(modelFile)).Scan(&id)
	if err != nil {
		log.Fatal(err)
	}

	log.Print("new best model to db ok. id = ", id)
}

func (r *Repo) getbest(ctx context.Context, args []string) {
	if len(args) < 1 {
		log.Fatal("not enough arguments for getbest")
	}

	bestFile := args[0]
	
	file, err := os.OpenFile(bestFile, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatal(err)
	}

	var name string
	err = r.db.QueryRow(ctx, "SELECT name FROM bestmodel WHERE train_id = $1", "").Scan(&name)
	if err != nil {
		log.Fatal(err)
	}

	_, err = file.WriteString(name)
	if err != nil {
		log.Fatal(err)
	}

	log.Print("write best model to ", bestFile)
}

