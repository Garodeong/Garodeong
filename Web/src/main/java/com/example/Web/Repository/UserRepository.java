package com.example.Web.Repository;

import com.example.Web.Domain.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {

    boolean existsByLoginId(String loginId);
    boolean existsByName(String name);
    Optional<User> findByLoginId(String loginId);
}